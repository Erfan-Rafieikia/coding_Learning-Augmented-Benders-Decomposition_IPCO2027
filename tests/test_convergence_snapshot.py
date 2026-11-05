"""Frozen-source and submit guard checks without Slurm or a Gurobi license."""
import json
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest

from openpyxl import Workbook

from scripts.verify_convergence_snapshot import snapshot_source, write_hashes, verify
from utils.exclusive_profiler import TIMING_SCHEMA_VERSION, SOLUTION_TIME_METRIC_VERSION
from utils.convergence_diagnostics import TRACE_SCHEMA_VERSION


class ConvergenceSnapshotTests(unittest.TestCase):
    def test_slurm_workbook_gate_accepts_current_schema_and_rejects_stale_or_failed_rows(self):
        source = Path(__file__).resolve().parents[1]
        wrapper = (source / 'my_job.pmedian_nearest.alliance.sbatch').read_text()
        gate = wrapper.split("<<'CHECK' || EXIT_CODE=$?\n", 1)[1].split('\nCHECK', 1)[0]
        current = dict(stopping_reason='OPTIMAL', timing_schema_version=TIMING_SCHEMA_VERSION,
                       solution_time_metric_version=SOLUTION_TIME_METRIC_VERSION,
                       convergence_trace_schema_version=TRACE_SCHEMA_VERSION)
        cases = [
            ('current', {}, True),
            ('legacy timing', {'timing_schema_version': 'exclusive_33_v1'}, False),
            ('missing timing', {'timing_schema_version': None}, False),
            ('legacy metric', {'solution_time_metric_version': 'inclusive_v1'}, False),
            ('missing metric', {'solution_time_metric_version': None}, False),
            ('legacy trace', {'convergence_trace_schema_version': 'retrospective_v1'}, False),
            ('sampled master trace', {'convergence_trace_schema_version': 'gurobi_master_v2'}, False),
            ('code error', {'stopping_reason': 'CODE_ERROR: fixture'}, False),
            ('null solution', {'stopping_reason': 'SOLUTION_NULL'}, False),
        ]
        with tempfile.TemporaryDirectory() as d:
            workbook = Path(d) / 'solution_log.xlsx'
            env = dict(os.environ, PYTHONPATH=str(source / 'src'))
            for name, changes, succeeds in cases:
                with self.subTest(case=name):
                    record = dict(current, **changes)
                    wb = Workbook()
                    wb.active.append(list(record))
                    wb.active.append(list(record.values()))
                    wb.save(workbook)
                    wb.close()
                    result = subprocess.run([sys.executable, '-', str(workbook)], input=gate,
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, succeeds, result.stderr)

    def test_frozen_source_survives_worktree_changes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            repo, root, data = (Path(d)/name for name in ('repo', 'pilot', 'data'))
            repo.mkdir(); root.mkdir(); data.mkdir()
            for name in ('src/main.py', 'scripts/a.py', 'tests/a.py', 'docs/a.md', 'my_job.pmedian_nearest.alliance.sbatch'):
                p = repo/name
                p.parent.mkdir(exist_ok=True)
                p.write_text('# original\n')
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c', 'user.email=test@example.com',
                            'commit', '-qm', 'fixture'], check=True)
            commit = snapshot_source(repo, root)
            (root/'configs').mkdir()
            for name in ('config.py', 'configs/task_0001.py', 'task_list.txt', 'pilot_manifest.tsv'):
                (root/name).write_text('fixture\n')
            (root/'source_data_root.txt').write_text(str(data)+'\n')
            (data/'instance.txt').write_text('input\n')
            write_hashes(root, data, ['instance.txt'])
            self.assertEqual(verify(root), commit)
            (repo/'src/main.py').write_text('# worktree moved\n')
            self.assertEqual(verify(root), commit)
            with self.assertRaisesRegex(ValueError, 'tracked changes'):
                snapshot_source(repo, root/'another')
            (root/'source/src/main.py').write_text('# tamper\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                verify(root)
            (root/'source/src/main.py').write_text('# original\n')
            (data/'instance.txt').write_text('changed input\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                verify(root)

    def test_real_dry_run_submits_nothing_and_rejects_excess_concurrency(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"MASTER_RUNS/labd_convergence_pilot_20261026_000000"
            data = Path(d)/"data"
            (root/"source/scripts").mkdir(parents=True)
            (root/"configs").mkdir(); data.mkdir()
            (root/"source/scripts/run_convergence_pilot_array.sbatch").write_text("# fixture")
            (root/"task_list.txt").write_text("configs/task_0001.py|SCFLP\n"*20)
            (root/"pilot_manifest.tsv").write_text("fixture")
            (root/"config.py").write_text("# fixture")
            (root/"source_commit.txt").write_text("0"*40+"\n")
            (root/"source_data_root.txt").write_text(str(data)+"\n")
            write_hashes(root, data, [])
            env = dict(os.environ, LABD_PYTHON=sys.executable, MAX_CONCURRENT="18")
            cmd = ["bash", str(source/"scripts/submit_convergence_pilots.sh"), str(root)]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DRY RUN ONLY", result.stdout)
            self.assertIn("128 GiB", result.stdout)
            self.assertIn("1-20%18", result.stdout)
            self.assertFalse((root/"submission.claim").exists())
            self.assertFalse((root/"job_id.txt").exists())
            env["MAX_CONCURRENT"] = "19"
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("1..18", result.stderr)


if __name__ == '__main__': unittest.main()
