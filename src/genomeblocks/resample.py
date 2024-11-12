from typing import Callable, Optional, List, Union, Dict, Tuple
import polars as pl
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection._split import _BaseKFold
import patsy


@dataclass
class SpatialBlockCV(_BaseKFold):
    """Spatial block cross-validation iterator compatible with BlockResampler.

    Parameters:
    -----------
    n_splits : int
        Number of folds for cross-validation
    block_size : float
        Size of spatial blocks in same units as input coordinates
    random_state : Optional[int]
        Random seed for reproducibility
    """

    n_splits: int = 5
    block_size: float = None
    random_state: Optional[int] = None

    def __post_init__(self):
        super().__init__(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

    def _assign_to_blocks(self, coords: np.ndarray) -> np.ndarray:
        """Assign points to blocks based on their coordinates"""
        x_min, y_min = coords[:, 0].min(), coords[:, 1].min()

        # Calculate block indices for each point
        x_idx = ((coords[:, 0] - x_min) / self.block_size).astype(int)
        y_idx = ((coords[:, 1] - y_min) / self.block_size).astype(int)

        # Create unique block identifier
        return x_idx * 1000000 + y_idx  # Large multiplier to avoid collisions

    def split(
        self, df: pl.DataFrame, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test sets.

        Parameters:
        -----------
        df : pl.DataFrame
            Input DataFrame
        coords : np.ndarray
            Array of shape (n_samples, 2) containing spatial coordinates

        Yields:
        -------
        train_idx, test_idx : Tuple[np.ndarray, np.ndarray]
            Indices for training and test sets
        """
        # Assign points to blocks
        block_assignments = self._assign_to_blocks(coords)
        unique_blocks = np.unique(block_assignments)

        # Randomly assign blocks to folds
        rng = np.random.RandomState(self.random_state)
        block_fold_map = {
            block: fold
            for block, fold in zip(
                unique_blocks, rng.randint(0, self.n_splits, size=len(unique_blocks))
            )
        }

        # Map blocks to folds for each point
        point_folds = np.array([block_fold_map[b] for b in block_assignments])

        # Generate train/test splits
        for fold in range(self.n_splits):
            test_mask = point_folds == fold
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]


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

    def spatial_cv(
        self,
        statistic_fn: Callable[[pl.DataFrame], Dict[str, float]],
        coords: np.ndarray,
        block_size: float,
        n_splits: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Perform spatial cross-validation

        Parameters:
        -----------
        statistic_fn : Callable
            Function that computes statistics on DataFrame
        coords : np.ndarray
            Array of shape (n_samples, 2) containing spatial coordinates
        block_size : float
            Size of spatial blocks
        n_splits : int
            Number of CV folds
        random_state : Optional[int]
            Random seed

        Returns:
        --------
        Dict containing cross-validation results for each statistic
        """
        cv = SpatialBlockCV(
            n_splits=n_splits, block_size=block_size, random_state=random_state
        )

        # Calculate statistics on full dataset
        full_stats = statistic_fn(self.df)

        # Perform CV
        cv_results = {var: [] for var in full_stats.keys()}

        for train_idx, test_idx in cv.split(self.df, coords):
            train_df = self.df.take(train_idx)
            test_df = self.df.take(test_idx)

            # Calculate statistics on test set
            test_stats = statistic_fn(test_df)

            for var in test_stats:
                cv_results[var].append(test_stats[var])

        # Compile results
        results = {}
        for var in full_stats:
            cv_values = np.array(cv_results[var])
            results[var] = {
                "full_sample": full_stats[var],
                "cv_estimates": cv_values,
                "cv_mean": np.mean(cv_values),
                "cv_std": np.std(cv_values),
                "cv_ci": np.percentile(cv_values, [2.5, 97.5]),
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
