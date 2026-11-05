"""Exclusive (non-overlapping) operation timings for LABD.

A ContextVar carries the currently active callback profiler across ordinary
function calls, so feature engineering / selection / parametric methods can
record detailed *nested* leaves without changing their public APIs.

Measured elapsed time is wall clock time. Exclusive parent durations subtract
all measured child spans; operation counts are independent of duration.
"""
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time
import math
import logging

TIMING_SCHEMA_VERSION = "exclusive_33_v2_final_only"
SOLUTION_TIME_METRIC_VERSION = "algorithm_excludes_final_reporting_v2"
CUT_CONSTRUCTION_TIMING_VERSION = "exact_ml_construction_v1"
CUT_ORIGINS = ("exact", "ml")

TIMING_CATEGORIES = (
    "input_data_load_validation",
    "scenario_generation_preprocessing",
    "master_model_construction",
    "phase0_solution_generation",
    "phase0_exact_subproblem_solving",
    "phase1_exact_subproblem_solving",
    "phase2_selected_exact_subproblem_solving",
    "phase2_fallback_exact_subproblem_solving",
    "standard_or_random_exact_subproblem_solving",
    "dual_trajectory_processing",
    "graph_random_walk_generation",
    "embedding_skipgram_training",
    "feature_scaling_transformation",
    "representative_selection_tree",
    "representative_selection_clustering",
    "representative_selection_pmedian",
    "representative_assignment_nearest_weighted",
    "rhs_projection_dimensionality_reduction",
    "regression_weight_training",
    "nonparametric_dual_transfer",
    "parametric_residual_prediction",
    "feasibility_check",
    "feasibility_repair_line_segment",
    "feasibility_repair_structure_aware",
    "objective_dominance_guard",
    "cuts_evaluation_generation",
    "cuts_insertion",
    "callback_other_logic",
    "master_solver_internal",
    "terminal_exact_original_objective_validation",
    "all_candidate_expost_reevaluation",  # Retired; reserved zero for old workbooks
    "diagnostics_logging_file_io",
    "remaining_unclassified_overhead",
)


# Operation labels identify a phase when one is intrinsic to the operation.
# Common operations span phases and are tagged 'multiple_callback_phases'
# rather than falsely assigned to one phase.
TIMING_PHASE_TAGS = {
    name: (
        "input" if name == "input_data_load_validation" else
        "preprocessing" if name == "scenario_generation_preprocessing" else
        "master_build" if name == "master_model_construction" else
        "phase0" if name.startswith("phase0_") else
        "phase1" if name.startswith("phase1_") else
        "phase2" if name.startswith("phase2_") else
        "postsolve" if name in {
            "terminal_exact_original_objective_validation",
            "all_candidate_expost_reevaluation",
        } else
        "master_solver" if name == "master_solver_internal" else
        "mixed_other" if name == "remaining_unclassified_overhead" else
        "multiple_callback_phases"
    )
    for name in TIMING_CATEGORIES
}

_ACTIVE_PROFILER = ContextVar("labd_active_exclusive_profiler", default=None)


class ExclusiveProfiler:
    def __init__(self, clock=None):
        self._clock = clock or time.perf_counter
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        # Detail spans are subsets of a category, never extra additive leaves.
        self.detail_totals = defaultdict(float)
        self.detail_counts = defaultdict(int)
        self._stack = []

    @contextmanager
    def measure(self, label, *, detail=None):
        label = str(label)
        frame = {"label": label, "started": self._clock(), "child": 0.0}
        token = _ACTIVE_PROFILER.set(self)
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = self._clock() - frame["started"]
            self._stack.pop()
            _ACTIVE_PROFILER.reset(token)
            exclusive = elapsed - frame["child"]
            if not math.isfinite(exclusive) or exclusive < -1e-9:
                raise RuntimeError("Invalid exclusive timer duration: " + label)
            exclusive = max(0.0, exclusive)
            self.totals[label] += exclusive
            self.counts[label] += 1
            if detail is not None:
                self.detail_totals[(label, detail)] += exclusive
                self.detail_counts[(label, detail)] += 1
            if self._stack:
                self._stack[-1]["child"] += elapsed

    def call(self, label, callable_, *args, **kwargs):
        with self.measure(label):
            return callable_(*args, **kwargs)

    def snapshot(self):
        return {k: float(v) for k, v in self.totals.items()}

    def snapshot_counts(self):
        return {k: int(v) for k, v in self.counts.items()}

    def cut_construction_breakdown(self):
        """Expression construction only; excludes evaluation and insertion.

        These details are already included in cuts_evaluation_generation.
        Counts are timed construction calls, not unique solver-retained cuts.
        """
        label = "cuts_evaluation_generation"
        return {
            **{f"{origin}_cut_construction_time_sec": float(
                self.detail_totals.get((label, origin), 0.0))
               for origin in CUT_ORIGINS},
            **{f"{origin}_cut_construction_count": int(
                self.detail_counts.get((label, origin), 0))
               for origin in CUT_ORIGINS},
            "cut_construction_timing_version": CUT_CONSTRUCTION_TIMING_VERSION,
        }

    def snapshot_schema(self):
        # All 33 categories have numerical zero values when not applicable.
        unknown = set(self.totals) - set(TIMING_CATEGORIES)
        if unknown:
            raise ValueError("Unknown timing categories: " + ", ".join(sorted(unknown)))
        return {key: float(self.totals.get(key, 0.0))
                for key in TIMING_CATEGORIES}

    def count_schema(self):
        return {key: int(self.counts.get(key, 0))
                for key in TIMING_CATEGORIES}


def reconcile_end_to_end(total, components, counts):
    """Validate measured spans and expose the unmeasured wall-time remainder.

    Counts are completed span entries (including exceptional exits). The two
    residual buckets count one accounted interval, not solver operations.
    This check never rescales measured durations to make a bad sum pass.
    """
    unknown = (set(components) | set(counts)) - set(TIMING_CATEGORIES)
    if unknown:
        raise ValueError("Unknown timing categories: " + ", ".join(sorted(unknown)))
    total = float(total)
    values = {k: float(components.get(k, 0.0)) for k in TIMING_CATEGORIES}
    operations = {k: counts.get(k, 0) for k in TIMING_CATEGORIES}
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid end-to-end duration")
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Nonfinite or negative measured duration")
    if any(type(v) is not int or v < 0 for v in operations.values()):
        raise ValueError("Invalid operation counts")
    remainder = total - math.fsum(v for k, v in values.items()
                                if k != "remaining_unclassified_overhead")
    if remainder < -max(1e-9, total * 1e-12):
        raise ValueError(f"Timing overlap: negative remainder {remainder:.9f}s")
    values["remaining_unclassified_overhead"] = max(0.0, remainder)
    operations["remaining_unclassified_overhead"] = 1
    for key in TIMING_CATEGORIES:
        if values[key] > 0 and operations[key] == 0:
            raise ValueError("Measured duration without an operation count: " + key)
    return values, operations


def solution_time_excluding_reporting(total, final_evaluation):
    """Exclude only the measured final-reporting interval from inclusive time."""
    total, final_evaluation = float(total), float(final_evaluation)
    if not all(math.isfinite(v) and v >= 0 for v in (total, final_evaluation)):
        raise ValueError("Invalid solution/reporting time")
    if final_evaluation > total:
        raise ValueError("Final reporting time exceeds total elapsed time")
    return total - final_evaluation


@contextmanager
def profile_section(label):
    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        yield
    else:
        with profiler.measure(label):
            yield


def profile_call(label, callable_, *args, **kwargs):
    with profile_section(label):
        return callable_(*args, **kwargs)


def profiled(label):
    """Decorate a function so it records a category ONLY inside a callback.

    On an ordinary call without an active callback profiler it is a no-op.
    """
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            return profile_call(label, func, *args, **kwargs)
        return wrapped
    return decorate


class _ProfiledLogger(logging.LoggerAdapter):
    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            with profile_section("diagnostics_logging_file_io"):
                super().log(level, msg, *args, **kwargs)


def get_profiled_logger(name):
    """Measure enabled in-callback logging; ordinary calls keep their behavior."""
    return _ProfiledLogger(logging.getLogger(name), {})
