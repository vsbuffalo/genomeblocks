import defopt
import polars as pl
import numpy as np
from typing import Literal, Optional, Dict, Union, List
from pathlib import Path
from diagnostics import run_diagnostics

from genomeblocks import GenomicBlocks, read_genomic_data
from resample import BlockResampler, calculate_statistics
from diagnostics import BlockDiagnostics, find_elbow_point


def format_results(
    method: str,
    var_results: Dict[str, Dict[str, Union[float, np.ndarray]]],
) -> None:
    """Format and print results for all variables"""
    print(f"\n{method} Results:")
    for var, results in var_results.items():
        print(f"\nVariable: {var}")
        print(f"Observed coefficient: {results['observed']:.4f}")
        print(f"95% CI: [{results['ci'][0]:.4f}, {results['ci'][1]:.4f}]")
        if "p_value" in results:
            p = results["p_value"]
            if p < 0.001:
                p_str = f"{p:.2e}"
            else:
                p_str = f"{p:.3f}"
            print(f"p-value: {p_str}")


def analyze(
    *,
    input_file: str,
    formula: str,
    method: Literal["block", "jack", "jack+"] = "block",
    block_size: int = 3,
    window_mode: bool = True,
    n_iterations: int = 1000,
    seed: Optional[int] = 42,
    show_progress: bool = True,
) -> None:
    """
    Analyze genomic data using block bootstrap.

    Args:
        input_file: Path to input file
        formula: R-style formula for analysis
        method: Analysis method ('block', 'jack', or 'jack+')
        block_size: Size of blocks for analysis
        window_mode: Whether to treat input as windows
        n_iterations: Number of bootstrap iterations
        seed: Random seed
        show_progress: Whether to show progress bar
    """
    blocks = read_genomic_data(input_file, block_size, window_mode)
    print(f"\nLoaded {len(blocks)} regions")
    blocked_df = blocks.create_blocks()
    resampler = BlockResampler(blocked_df)

    stat_fn = lambda df: calculate_statistics(df, formula)
    if method == "block":
        results = resampler.bootstrap(
            statistic_fn=stat_fn,
            n_iterations=n_iterations,
            seed=seed,
            show_progress=show_progress,
        )

    format_results(method, results)


def diagnose(
    *,
    input_file: str,
    formula: str,
    window_mode: bool = True,
    min_block_size: int = 1,
    max_block_size: int = 20,
    step: int = 1,
    n_iterations: int = 1000,
    seed: Optional[int] = 42,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run block bootstrap diagnostics to analyze optimal block size.

    Args:
        input_file: Path to input file
        formula: R-style formula for analysis
        window_mode: Whether to treat input as windows
        min_block_size: Minimum block size to test
        max_block_size: Maximum block size to test
        step: Step size for block sizes
        n_iterations: Number of bootstrap iterations
        seed: Random seed
        output_dir: Directory to save diagnostic plots and results (optional)
    """
    import matplotlib.pyplot as plt

    print("\nRunning block size diagnostics...")

    # Run diagnostics
    diagnostic_results = run_diagnostics(
        input_file=input_file,
        formula=formula,
        window_mode=window_mode,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        step=step,
        n_iterations=n_iterations,
        seed=seed,
    )

    # Save results if output directory is specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save plot
        plt.savefig(output_path / "block_size_diagnostics.png")
        print(
            f"\nSaved diagnostic plot to {output_path / 'block_size_diagnostics.png'}"
        )

        # Save numerical results
        results_df = []

        for var, size_results in diagnostic_results.items():
            var_df = {
                "variable": [var] * len(size_results),
                "block_size": list(size_results.keys()),
                "std_error": list(size_results.values()),
            }
            results_df.append(pl.DataFrame(var_df))

        results_df = pl.concat(results_df)

        results_df.write_csv(output_path / "block_size_results.tsv", separator="\t")
        print(f"Saved numerical results to {output_path / 'block_size_results.tsv'}")

    # Print suggestions
    print("\nDiagnostic Analysis:")
    for var in diagnostic_results:
        block_sizes = np.array(list(diagnostic_results[var].keys()))
        stderrs = np.array(list(diagnostic_results[var].values()))
        suggested_size = find_elbow_point(block_sizes, stderrs)
        print(f"\n{var}:")
        print(f"  Suggested block size: {suggested_size}")
        print(f"  Standard error range: {min(stderrs):.4f} - {max(stderrs):.4f}")


def main():
    """Command line interface for genomeblocks analysis"""
    commands = {"analyze": analyze, "diagnose": diagnose}
    defopt.run(commands, show_defaults=True)


if __name__ == "__main__":
    main()
