"""
Estimated runtimes for notebook tests (in seconds).
Based on sequential execution measurements.
Used for Longest Job First (LJF) and optimal bin-packing scheduling.

To update these values, run the sequential tests and extract timing info:
    python -m unittest discover -s . -p "tb_*.py" -v

Tests are sorted by runtime (longest first) for optimal load balancing.
"""

import heapq
import os
import pandas as pd

# Estimated runtimes in seconds (from sequential run on 2025-12-01)
# Sorted from longest to shortest for easy reference
# These are fallback values if CSV file is not available
TEST_RUNTIMES_FALLBACK = {
    # Long-running tests (> 60s) - these are the bottleneck
    "tb_iris": 138.10,
    "tb_elliptic_pde": 96.12,
    "tb_qei_demo_for_blog": 88.40,
    "tb_asian_option_mlqmc": 81.97,
    "tb_vectorized_qmc_bayes": 69.83,
    # Medium-running tests (20-60s)
    "tb_digital_net_b2": 41.37,
    "tb_ray_tracing": 38.57,
    "tb_gaussian_diagnostics_demo": 35.80,
    # Short-running tests (< 20s)
    "tb_sorokin_thesis_2025": 19.20,
    "tb_lattice_random_generator": 18.54,
    "tb_linear_scrambled_halton": 17.88,
    "tb_MCQMC_2020_QMC_Software_Tutorial": 16.94,
    "tb_vectorized_qmc": 16.89,
    "tb_plot_proj_function": 14.29,
    "tb_joss2025": 12.93,
    "tb_Iteration_Log_Tolerance_Demo": 12.37,
    "tb_gbm_examples": 11.43,
    "tb_accuracy_and_resume": 11.09,
    "tb_resume_examples": 10.97,
    "tb_control_variates": 10.81,
    "tb_pricing_options": 10.43,
    "tb_joss2026": 9.46,
    "tb_scipywrapper_demo": 9.30,
    "tb_sample_scatter_plots": 8.16,
    "tb_nei_demo": 7.58,
    "tb_some_true_measures": 7.18,
    "tb_copula_examples": 5.68,
    "tb_why_add_q_to_mc_blog": 4.73,
    "tb_acceptance_rejection": 4.54,
    "tb_lebesgue_integration": 4.10,
    "tb_qmcpy_intro": 4.01,
    "tb_quickstart": 3.81,
    "tb_gbm_demo": 1.82,
    "tb_Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026": 1.76,
    "tb_umbridge": 0.85,
    # Skipped tests (assigned small default time)
    "tb_Argonne_2023_Talk_Figures": 1.0,
    "tb_MCQMC2022_Article_Figures": 1.0,
    "tb_pydata_chi_2023": 1.0,
}

# Default runtime for unknown tests (assume medium-length)
DEFAULT_RUNTIME = 30.0

# Try to load runtimes from CSV file
TEST_RUNTIMES = {}
csv_path = os.path.join(
    os.path.dirname(__file__),
    "../../demos/talk_paper_demos/Parslfest_2025/output/sequential_output_time.csv",
)
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Convert from test_name to tb_name format
        for _, row in df.iterrows():
            notebook = row["Notebook"]
            time_s = float(row["Time_s"])
            # Convert test_xxx to tb_xxx format
            tb_name = notebook.replace("test_", "tb_")
            TEST_RUNTIMES[tb_name] = time_s
        print(f"Loaded {len(TEST_RUNTIMES)} test runtimes from {csv_path}")
    except Exception as e:
        print(f"Warning: Failed to load runtimes from CSV: {e}")
        TEST_RUNTIMES = TEST_RUNTIMES_FALLBACK.copy()
else:
    # Fall back to hardcoded values
    TEST_RUNTIMES = TEST_RUNTIMES_FALLBACK.copy()


def get_runtime(test_name: str) -> float:
    """Get estimated runtime for a test module."""
    return TEST_RUNTIMES.get(test_name, DEFAULT_RUNTIME)


def optimal_schedule(test_modules: list, num_workers: int) -> tuple:
    """
    Compute optimal job scheduling using LPT (Longest Processing Time) algorithm.
    This is a greedy bin-packing approach that assigns each job to the least loaded worker.

    Args:
        test_modules: List of test module names
        num_workers: Number of parallel workers

    Returns:
        tuple: (ordered_modules, worker_assignments, estimated_makespan)
            - ordered_modules: Tests ordered for optimal submission
            - worker_assignments: List of (worker_id, [test_names])
            - estimated_makespan: Estimated parallel execution time
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    # Sort by runtime (longest first) - LPT heuristic
    sorted_modules = sorted(test_modules, key=lambda m: get_runtime(m), reverse=True)

    # Min-heap of (current_load, worker_id, [assigned_tests])
    workers = [(0.0, i, []) for i in range(num_workers)]
    heapq.heapify(workers)

    # Assign each job to the least loaded worker
    for module in sorted_modules:
        runtime = get_runtime(module)
        # Pop the least loaded worker
        current_load, worker_id, assigned = heapq.heappop(workers)
        # Assign this job
        assigned.append(module)
        new_load = current_load + runtime
        # Push back with updated load
        heapq.heappush(workers, (new_load, worker_id, assigned))

    # Extract results
    worker_assignments = [
        (w_id, tests) for (load, w_id, tests) in sorted(workers, key=lambda x: x[1])
    ]
    worker_loads = [
        (w_id, sum(get_runtime(t) for t in tests))
        for (w_id, tests) in worker_assignments
    ]
    estimated_makespan = max(load for (_, load) in worker_loads) if worker_loads else 0

    # Create optimal submission order: interleave from each worker's queue
    # to maximize parallelism from the start
    ordered_modules = []
    queues = [list(tests) for (_, tests) in worker_assignments]
    while any(queues):
        for q in queues:
            if q:
                ordered_modules.append(q.pop(0))

    return ordered_modules, worker_assignments, estimated_makespan
