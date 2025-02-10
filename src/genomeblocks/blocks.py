import polars as pl
from dataclasses import dataclass


@dataclass
class GenomicBlocks:
    """Class to handle genomic blocks for permutation testing"""

    df: pl.DataFrame
    block_size: int
    window_mode: bool = True  # True if input rows are windows, False if they are points

    def __post_init__(self):
        required_cols = ["chrom", "start", "end"]
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Ensure chromosome is string type
        self.df = self.df.with_columns(pl.col("chrom").cast(pl.Utf8))

    def __len__(self) -> int:
        """Return number of regions in the genomic blocks"""
        return len(self.df)

    def create_blocks(self) -> pl.DataFrame:
        """Create blocks based on mode and block size"""
        if self.window_mode:
            return self._create_blocks_from_windows()
        else:
            return self._create_blocks_from_points()

    def _create_blocks_from_windows(self) -> pl.DataFrame:
        """Create blocks by grouping adjacent windows"""
        return (
            self.df.with_columns(
                [
                    # Create block IDs within each chromosome
                    pl.col("chrom")
                    .repeat_by(1)
                    .cum_count()
                    .over("chrom")
                    .floordiv(self.block_size)
                    .alias("block_id")
                ]
            )
            # Combine original chrom and block_id for unique blocks
            .with_columns(
                [
                    pl.concat_str(
                        [pl.col("chrom"), pl.lit("_"), pl.col("block_id").cast(pl.Utf8)]
                    ).alias("block_identifier")
                ]
            )
        )

    def _create_blocks_from_points(self) -> pl.DataFrame:
        """Create blocks by binning points"""
        return (
            self.df.sort(["chrom", "start"])
            .with_columns(
                [
                    # First get chromosome-specific counts
                    pl.col("chrom")
                    .repeat_by(1)
                    .cum_count()
                    .over("chrom")
                    .alias("point_index"),
                    # Calculate block boundaries
                    pl.col("start").min().over("chrom").alias("chrom_start"),
                    pl.col("end").max().over("chrom").alias("chrom_end"),
                ]
            )
            .with_columns(
                [
                    # Create blocks within each chromosome
                    (
                        (pl.col("start") - pl.col("chrom_start"))
                        / (
                            (pl.col("chrom_end") - pl.col("chrom_start"))
                            / (
                                pl.col("point_index").max().over("chrom")
                                / self.block_size
                            ).floor()
                        )
                    )
                    .floor()
                    .alias("block_id")
                ]
            )
            .with_columns(
                [
                    pl.concat_str(
                        [pl.col("chrom"), pl.lit("_"), pl.col("block_id").cast(pl.Utf8)]
                    ).alias("block_identifier")
                ]
            )
            .drop(["point_index", "chrom_start", "chrom_end"])
        )


def read_genomic_data(
    filepath: str, block_size: int, window_mode: bool = True, **kwargs
) -> GenomicBlocks:
    """
    Read genomic data from TSV file and create GenomicBlocks object

    Parameters:
    -----------
    filepath : str
        Path to TSV file
    block_size : int
        Number of windows/regions to group into a block
    window_mode : bool
        If True, treat rows as windows. If False, treat rows as points
    **kwargs : dict
        Additional arguments passed to polars.read_csv

    Returns:
    --------
    GenomicBlocks
        Processed genomic blocks object
    """
    df = pl.read_csv(filepath, separator="\t", **kwargs)

    return GenomicBlocks(df, block_size, window_mode)
