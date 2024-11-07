import polars as pl
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from genomeblocks import GenomicBlocks
from resample import BlockResampler, calculate_statistics
from tqdm.auto import tqdm


@dataclass
class BlockDiagnostics:
    """Class to handle block bootstrap diagnostics"""

    df: pl.DataFrame
    formula: str
    window_mode: bool = True
    min_block_size: int = 2
    max_block_size: int = 20
    step: int = 2
    n_iterations: int = 1000
    seed: Optional[int] = 42

    def analyze_block_sizes(
        self, show_progress: bool = True
    ) -> Dict[str, Dict[int, float]]:
        """Analyze how different block sizes affect standard errors"""
        block_sizes = range(self.min_block_size, self.max_block_size + 1, self.step)
        results = {}
        first_run = True

        # Create progress bar
        pbar = tqdm(
            block_sizes, desc="Analyzing block sizes", disable=not show_progress
        )

        for block_size in pbar:
            # Create blocks for this size
            blocks = GenomicBlocks(self.df, block_size, self.window_mode)
            blocked_df = blocks.create_blocks()
            resampler = BlockResampler(blocked_df)

            # Run bootstrap
            bootstrap_results = resampler.bootstrap(
                statistic_fn=lambda df: calculate_statistics(df, self.formula),
                n_iterations=self.n_iterations,
                seed=self.seed,
                show_progress=False,  # Disable inner progress bar
            )

            # Store standard errors for each variable
            for var in bootstrap_results:
                if var not in results:
                    results[var] = {}
                results[var][block_size] = np.std(
                    bootstrap_results[var]["bootstrap_estimates"]
                )

                # Update progress bar description with current variable
                if first_run:
                    pbar.set_description(f"Analyzing block sizes for {var}")
            first_run = False

        return results

    def plot_block_size_analysis(self, results: Dict[str, Dict[int, float]]) -> None:
        """Create diagnostic plot for block size analysis"""
        plt.figure(figsize=(10, 6))

        for var in results:
            block_sizes = list(results[var].keys())
            stderrs = list(results[var].values())
            plt.plot(block_sizes, stderrs, "o-", label=var)

        plt.xlabel("Block Size")
        plt.ylabel("Bootstrap Standard Error")
        plt.title("Block Size vs Standard Error")
        plt.legend()
        plt.grid(True)

        # Add elbow analysis suggestion
        for var in results:
            block_sizes = np.array(list(results[var].keys()))
            stderrs = np.array(list(results[var].values()))
            suggested_size = find_elbow_point(block_sizes, stderrs)
            plt.axvline(x=suggested_size, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()


def find_elbow_point(x: np.ndarray, y: np.ndarray) -> int:
    """Find the elbow point in the curve using the elbow method"""
    # Normalize the data
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Find point furthest from line between first and last points
    coords = np.vstack((x_norm, y_norm)).T
    first_point = coords[0]
    last_point = coords[-1]
    line_vec = last_point - first_point

    # Vector from first point to each point
    point_vec = coords - first_point

    # Distance from each point to the line
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec * line_len

    # Find perpendicular distance
    distances = np.cross(point_vec_scaled, line_unitvec)
    elbow_idx = np.argmax(np.abs(distances))

    return x[elbow_idx]


def run_diagnostics(
    input_file: str,
    formula: str,
    window_mode: bool = True,
    min_block_size: int = 2,
    max_block_size: int = 20,
    step: int = 2,
    n_iterations: int = 1000,
    seed: Optional[int] = 42,
    show_progress: bool = True,
) -> Tuple[Dict[str, Dict[int, float]], plt.Figure]:
    """Run block bootstrap diagnostics and return results"""
    df = pl.read_csv(input_file, separator="\t")
    diagnostics = BlockDiagnostics(
        df=df,
        formula=formula,
        window_mode=window_mode,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        step=step,
        n_iterations=n_iterations,
        seed=seed,
    )

    results = diagnostics.analyze_block_sizes(show_progress=show_progress)
    diagnostics.plot_block_size_analysis(results)

    return results
