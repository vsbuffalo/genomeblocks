# GenomeBlocks

GenomeBlocks is a Python package for performing block bootstrap analysis on genomic data. It provides tools for analyzing spatial correlations and statistical significance in genomic regions while accounting for spatial dependencies.

## Features

- Block bootstrap analysis of genomic regions
- Support for both window-based and point-based genomic data
- Parallel processing for efficient bootstrap iterations
- Diagnostic tools for optimal block size selection
- R-style formula interface for statistical modeling
- Comprehensive statistical output including confidence intervals and p-values

<!--
## Installation

```bash
pip install genomeblocks
```
-->

## Command Line Interface

GenomeBlocks provides two main commands: `resample` for performing block
bootstrap analysis and `diagnose` for determining optimal block sizes. The CLI
only performs simple linear regression; for bootstraps of more complicated
statistical functions, use the `genomeblocks` library.

### Block Bootstrap

Below is a simple example of a genomic linear regression with block
bootstrapped CIs and slope p-values (against β₀ = 0).

```bash
genomeblocks resample \
    --input-file data.tsv \
    --formula "y ~ x1 + x2" \
    --method block \
    --block-size 3 \
    --n-iterations 1000
```

### Block Size Diagnostics

This runs the block bootstrap procedure over varying block sizes, which can be
used to identify where the variance is the highest. Assuming positive spatial
covariance (neighboring blocks are more alike than distant ones), this can be
used to to find the scale of across-block covariance.

```bash
genomeblocks diagnose \
    --input-file data.tsv \
    --formula "y ~ x1 + x2" \
    --min-block-size 1 \
    --max-block-size 20 \
    --output-dir results/
```

## Python Library Usage

### Basic Usage

```python
from genomeblocks import read_genomic_data
from resample import BlockResampler, calculate_statistics

# Read and process genomic data
blocks = read_genomic_data("data.tsv", block_size=3, window_mode=True)
blocked_df = blocks.create_blocks()

# Create resampler
resampler = BlockResampler(blocked_df)

# Define statistical analysis
formula = "y ~ x1 + x2"
stat_fn = lambda df: calculate_statistics(df, formula)

# Perform bootstrap analysis
results = resampler.bootstrap(
    statistic_fn=stat_fn,
    n_iterations=1000,
    seed=42,
    show_progress=True
)

# Access results
for var, res in results.items():
    print(f"\nVariable: {var}")
    print(f"Coefficient: {res['observed']:.4f}")
    print(f"95% CI: [{res['ci'][0]:.4f}, {res['ci'][1]:.4f}]")
    print(f"p-value: {res['p_value']:.4f}")
```

### Custom Statistics

You can define custom statistics by creating a function that takes a Polars DataFrame and returns a dictionary of results:

```python
def custom_statistic(df: pl.DataFrame) -> dict:
    # Your custom statistical calculation here
    return {"statistic_name": result_value}

results = resampler.bootstrap(
    statistic_fn=custom_statistic,
    n_iterations=1000
)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
