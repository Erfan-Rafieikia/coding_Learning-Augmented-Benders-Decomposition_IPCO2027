"""
run_registry.py

Utility module for managing unique run identifiers and directory structures across
both local and remote (SLURM) execution environments.

This module provides:
1. Unique identifier generation (SLURM_JOB_ID for HPC, UUID for local)
2. Problem class detection from configuration
3. Run directory creation with consistent structure: {PROBLEM_CLASS}_class/RESULTS/{UniqueID}/
4. Config snapshot saving (submit.sh, my_job.sbatch, config.py content)
5. Cross-platform compatibility (Windows, Linux, macOS)
"""

import os
import sys
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


def get_unique_identifier() -> str:
    """
    Generate a unique identifier for this run.
    
    For SLURM jobs: Returns SLURM_JOB_ID (cluster execution)
    For local runs: Returns UUID-based identifier
    
    Returns:
        str: Unique identifier (e.g., "12345678" for SLURM or "abc123def-uuid4" for local)
    """
    if 'SLURM_JOB_ID' in os.environ:
        # Running on SLURM cluster - use job ID
        return str(os.environ['SLURM_JOB_ID'])
    else:
        # Local execution - generate UUID
        return str(uuid.uuid4())[:8]  # Shortened UUID for cleaner folder names


def get_timestamp_str() -> str:
    """
    Generate a timestamp string in format: YYYYMMDD_HHMMSS
    
    Returns:
        str: Timestamp string (e.g., "20260329_143022")
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_execution_mode() -> str:
    """
    Determine execution mode: 'slurm', 'local'
    
    Returns:
        str: Execution mode
    """
    return 'slurm' if 'SLURM_JOB_ID' in os.environ else 'local'


def get_problem_class_from_config(config_module=None) -> str:
    """
    Detect the problem class from the configuration.
    
    Since user specifies only ONE problem class per run, this checks
    which class has non-empty instances in problem_classes_to_instances.
    
    Args:
        config_module: Optional imported config module. If None, attempts to import.
    
    Returns:
        str: Problem class name (SCFLP, SUMND, SCMND, or default SCFLP)
    """
    if config_module is None:
        try:
            import config
            config_module = config
        except ImportError:
            return "SCFLP"  # Default fallback
    
    # Try to get problem_classes_to_instances
    if hasattr(config_module, 'problem_classes_to_instances'):
        pci = config_module.problem_classes_to_instances
        # Find the first non-empty problem class
        for problem_class in ['SCFLP', 'SUMND', 'SCMND']:
            if problem_class in pci and pci[problem_class]:
                return problem_class
    
    return "SCFLP"  # Default fallback


def create_run_directory(problem_class: str, project_root: str) -> str:
    """
    Create the run directory structure:
    {PROJECT_ROOT}/RESULTS/{PROBLEM_CLASS}_class/RESULTS/{UniqueID}_{Timestamp}/
    
    Args:
        problem_class: Problem class (SCFLP, SUMND, SCMND)
        project_root: Root directory of the project
    
    Returns:
        str: Full path to the created run directory
    """
    unique_id = get_unique_identifier()
    timestamp_str = get_timestamp_str()
    run_folder_name = f"{unique_id}_{timestamp_str}"
    
    run_dir = os.path.join(
        project_root, "RESULTS", f"{problem_class}_class", "RESULTS", run_folder_name
    )
    
    # Create directory with parents
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def save_config_snapshot(
    run_dir: str, 
    project_root: str,
    snapshot_info: Optional[dict] = None
) -> None:
    """
    Save snapshots of configuration files to the run directory.
    
    Saves:
    1. submit.sh (if exists)
    2. my_job.sbatch (if exists)
    3. config.py (if exists)
    4. run_metadata.json (execution info)
    
    Args:
        run_dir: Run directory path
        project_root: Project root path
        snapshot_info: Optional dict with additional metadata to save
    """
    os.makedirs(run_dir, exist_ok=True)
    
    # Files to snapshot
    config_files = [
        ('submit.sh', 'submit.sh'),
        ('my_job.sbatch', 'my_job.sbatch'),
        ('src/config.py', 'config.py'),
    ]
    
    for source_file, snapshot_name in config_files:
        source_path = os.path.join(project_root, source_file)
        if os.path.exists(source_path):
            snapshot_path = os.path.join(run_dir, f"snapshot_{snapshot_name}")
            try:
                with open(source_path, 'r') as f:
                    content = f.read()
                with open(snapshot_path, 'w') as f:
                    f.write(content)
            except Exception as e:
                print(f"Warning: Could not snapshot {source_file}: {e}", file=sys.stderr)
    
    # Save metadata
    metadata = {
        'unique_identifier': get_unique_identifier(),
        'timestamp': get_timestamp_str(),
        'execution_mode': get_execution_mode(),
        'problem_class': snapshot_info.get('problem_class', 'UNKNOWN') if snapshot_info else 'UNKNOWN',
        'project_root': project_root,
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', None),
        'hostname': os.environ.get('HOSTNAME', 'unknown'),
    }
    
    if snapshot_info:
        metadata.update(snapshot_info)
    
    metadata_path = os.path.join(run_dir, 'run_metadata.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save metadata: {e}", file=sys.stderr)


def get_run_identifier_display() -> str:
    """
    Get a user-friendly display string for the run identifier.
    
    Returns:
        str: Display string (e.g., "[RUN: HPC-12345678 | 2026-03-29 14:30:22]")
    """
    unique_id = get_unique_identifier()
    timestamp_str = get_timestamp_str()
    mode = get_execution_mode().upper()
    
    if mode == 'SLURM':
        mode_str = f"HPC-{unique_id}"
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            mode_str += f"[Task-{os.environ['SLURM_ARRAY_TASK_ID']}]"
    else:
        mode_str = f"LOCAL-{unique_id}"
    
    # Format timestamp for display
    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return f"[RUN: {mode_str} | {display_time}]"


if __name__ == "__main__":
    # Simple test
    print(f"Unique Identifier: {get_unique_identifier()}")
    print(f"Timestamp: {get_timestamp_str()}")
    print(f"Execution Mode: {get_execution_mode()}")
    print(f"Display: {get_run_identifier_display()}")
