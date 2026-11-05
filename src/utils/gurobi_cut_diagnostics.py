"""Solver-reported cut counts, distinct from callback submission counts.

MIP_CUTCNT is an aggregate current count and may decrease. The log summary
describes the last LP, not all cuts ever accepted. Neither identifies exact
versus ML origins. Missing log output is unavailable, not a zero count.
"""
import json
import math
import re

CUT_COUNT_SCHEMA_VERSION = "gurobi_cuts_v1"
CUT_COUNT_FIELDS = (
    "gurobi_cut_count_schema_version",
    "gurobi_mip_cut_count_last_observed",
    "gurobi_mip_cut_count_peak_observed",
    "gurobi_mip_cut_count_last_observed_time_sec",
    "gurobi_mip_cut_count_observation_count",
    "gurobi_last_lp_cut_summary_status",
    "gurobi_last_lp_cut_counts_json",
    "gurobi_last_lp_total_cut_count",
    "gurobi_last_lp_user_cut_count",
    "gurobi_last_lp_lazy_constraint_count",
)


class GurobiCutStatistics:
    def __init__(self):
        self.last_count = None
        self.peak_count = None
        self.last_time = None
        self.observation_count = 0
        self._buffer = ""
        self._in_section = False
        self._section_counts = None
        self._section_complete = False

    def observe(self, count, elapsed):
        count, elapsed = float(count), float(elapsed)
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError("Invalid Gurobi MIP_CUTCNT")
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid Gurobi cut-count timestamp")
        count = int(count)
        changed = count != self.last_count
        self.last_count = count
        self.peak_count = count if self.peak_count is None else max(self.peak_count, count)
        self.last_time = elapsed
        self.observation_count += 1
        return changed

    def feed_message(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line):
        if line.strip() == "Cutting planes:":
            self._in_section = True
            self._section_counts = {}
            self._section_complete = False
        elif self._in_section:
            match = re.fullmatch(r"\s+([^:]+):\s*(\d+)\s*", line)
            if match:
                self._section_counts[match[1].strip()] = int(match[2])
            elif not line.strip() or line.lstrip().startswith("Explored "):
                self._in_section = False
                self._section_complete = True
            else:
                # Preserve an explicit unavailable/incomplete status if the
                # log format changes rather than inventing zero counts.
                self._in_section = False
                self._section_complete = False

    def finalize(self):
        if self._buffer:
            self._consume_line(self._buffer.rstrip("\r"))
            self._buffer = ""

    def export(self):
        counts = self._section_counts if self._section_complete else None
        status = ("reported" if counts is not None else
                  "incomplete" if self._section_counts is not None else "not_reported")
        return {
            "gurobi_cut_count_schema_version": CUT_COUNT_SCHEMA_VERSION,
            "gurobi_mip_cut_count_last_observed": self.last_count,
            "gurobi_mip_cut_count_peak_observed": self.peak_count,
            "gurobi_mip_cut_count_last_observed_time_sec": self.last_time,
            "gurobi_mip_cut_count_observation_count": self.observation_count,
            "gurobi_last_lp_cut_summary_status": status,
            "gurobi_last_lp_cut_counts_json": json.dumps(counts, sort_keys=True) if counts is not None else None,
            "gurobi_last_lp_total_cut_count": sum(counts.values()) if counts is not None else None,
            "gurobi_last_lp_user_cut_count": counts.get("User", 0) if counts is not None else None,
            "gurobi_last_lp_lazy_constraint_count": counts.get("Lazy constraints", 0) if counts is not None else None,
        }
