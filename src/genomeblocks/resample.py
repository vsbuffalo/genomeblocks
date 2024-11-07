from typing import Callable, Optional, Tuple, List, Union, Dict
import polars as pl
import numpy as np
from dataclasses import dataclass
import warnings
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from functools import partial
from scipy import stats
from sklearn.linear_model import LinearRegression
import patsy


@dataclass
class BlockResampler:
    """Class to handle block bootstrapping and jackknifing with parallel processing"""

    df: pl.DataFrame
    block_identifier: str = "block_identifier"
    n_jobs: int = -1  # -1 means use all available cores

    def __post_init__(self):
        if self.block_identifier not in self.df.columns:
            raise ValueError(f"DataFrame must contain column: {self.block_identifier}")

        self.unique_blocks = self.df[self.block_identifier].unique().to_list()
        self.n_blocks = len(self.unique_blocks)

    def _bootstrap_iteration(
        self, statistic_fn: Callable[[pl.DataFrame], float], seed: Optional[int] = None
    ) -> float:
        """Single bootstrap iteration for parallel processing"""
        if seed is not None:
            np.random.seed(seed)

        # Sample blocks with replacement
        sampled_blocks = np.random.choice(
            self.unique_blocks, size=self.n_blocks, replace=True
        )

        # Create bootstrap sample
        bootstrap_df = pl.concat(
            [
                self.df.filter(pl.col(self.block_identifier) == block)
                for block in sampled_blocks
            ]
        )

        return statistic_fn(bootstrap_df)

    def bootstrap(
        self,
        statistic_fn: Callable[[pl.DataFrame], Dict[str, float]],
        n_iterations: int = 1000,
        seed: Optional[int] = None,
        show_progress: bool = True,
        null_value: float = 0.0,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Perform bootstrap for multiple statistics"""
        observed_stats = statistic_fn(self.df)

        # Generate bootstrap samples
        if seed is not None:
            np.random.seed(seed)
        seeds = np.random.randint(0, 2**32, size=n_iterations)

        bootstrap_iterations = Parallel(n_jobs=self.n_jobs)(
            delayed(self._bootstrap_iteration)(statistic_fn, seed_i)
            for seed_i in tqdm(seeds, disable=not show_progress)
        )

        results = {}
        for var in observed_stats:
            bootstrap_values = np.array(
                [iter_stats[var] for iter_stats in bootstrap_iterations]
            )

            # Two-sided p-value that properly handles non-symmetric distributions
            p_left = np.mean(bootstrap_values <= null_value)
            p_right = np.mean(bootstrap_values >= null_value)
            p_value = 2 * min(p_left, p_right)

            results[var] = {
                "observed": observed_stats[var],
                "bootstrap_estimates": bootstrap_values,
                "ci": np.percentile(bootstrap_values, [2.5, 97.5]),
                "p_value": p_value,
            }

        return results


def parse_formula(
    formula: str, df: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Parse R-like formula and return X and y arrays plus variable names"""
    pdf = df.to_pandas()
    y, X = patsy.dmatrices(formula, data=pdf, return_type="dataframe")
    # Get variable names excluding intercept
    var_names = (
        X.columns.tolist()[1:] if "Intercept" in X.columns else X.columns.tolist()
    )
    return X.to_numpy(), y.to_numpy().ravel(), var_names


def calculate_statistics(df: pl.DataFrame, formula: str) -> Dict[str, float]:
    """Calculate regression coefficients for all variables in formula"""
    X, y, var_names = parse_formula(formula, df)

    if X.shape[1] == 1:  # Simple regression
        slope, _, _, _, _ = stats.linregress(X.ravel(), y)
        return {var_names[0]: slope}
    else:  # Multiple regression
        model = LinearRegression(fit_intercept=False).fit(X, y)
        # If there's an intercept, it's in the design matrix but not in var_names
        # So we want the full model.coef_ to match with var_names
        coefficients = {
            name: float(coef)  # Ensure we're getting actual floats
            for name, coef in zip(var_names, model.coef_)
        }
        return coefficients
