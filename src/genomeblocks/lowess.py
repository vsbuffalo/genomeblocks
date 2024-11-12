from typing import Callable, Optional, List, Union, Dict, Tuple
import polars as pl
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection._split import _BaseKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from genomeblocks import GenomicBlocks


class BlockCV(_BaseKFold):
    """Cross-validation iterator that respects block structure.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds for cross-validation
    block_col : str
        Name of column containing block identifiers
    random_state : Optional[int]
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        block_col: str = "block_identifier",
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.block_col = block_col
        self.df = None
    
    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test sets."""
        if self.df is None:
            raise ValueError("Must call set_df before splitting")
            
        # Get unique blocks
        unique_blocks = self.df[self.block_col].unique().to_list()
        
        # Randomly assign blocks to folds
        rng = np.random.RandomState(self.random_state)
        block_fold_map = {
            block: fold for block, fold in zip(
                unique_blocks, 
                rng.randint(0, self.n_splits, size=len(unique_blocks))
            )
        }
        
        # Map blocks to folds for each point
        point_folds = np.array([
            block_fold_map[block] for block in self.df[self.block_col].to_list()
        ])
        
        # Generate train/test splits
        for fold in range(self.n_splits):
            test_mask = point_folds == fold
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits
    
    def set_df(self, df: pl.DataFrame) -> 'BlockCV':
        """Set the DataFrame containing block information."""
        self.df = df
        return self

class LOWESSRegressor(BaseEstimator, RegressorMixin):
    """LOWESS regression compatible with scikit-learn API."""
    
    def __init__(self, frac: float = 0.5):
        self.frac = frac
    
    def fit(self, X, y):
        if X.ndim > 1:
            X = X.ravel()
        self.X_train_ = X
        self.y_train_ = y
        return self
    
    def predict(self, X):
        if X.ndim > 1:
            X = X.ravel()
        return lowess(self.y_train_, self.X_train_, xvals=X, frac=self.frac)

def optimize_lowess(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    block_col: str = "block_identifier",
    param_grid: Optional[dict] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> tuple[LOWESSRegressor, dict]:
    """Find optimal LOWESS parameters using block-aware cross-validation."""
    if param_grid is None:
        param_grid = {
            'frac': np.linspace(0.1, 0.9, 9)
        }
    
    # Create and configure CV splitter
    cv = BlockCV(n_splits=n_splits, block_col=block_col, random_state=random_state)
    cv.set_df(df)  # Set the DataFrame for block information
    
    # Setup the grid search
    lowess_reg = LOWESSRegressor()
    grid_search = GridSearchCV(
        estimator=lowess_reg,
        param_grid=param_grid,
        cv=cv,  # Pass the CV splitter directly
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )
    
    # Fit the grid search
    X = df[x_col].to_numpy().reshape(-1, 1)
    y = df[y_col].to_numpy()
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.cv_results_



def process_lowess_cv_results(cv_results: Dict) -> Tuple[float, pl.DataFrame]:
    """Process GridSearchCV results for LOWESS into a more readable format.
    
    Parameters:
    -----------
    cv_results : Dict
        cv_results_ from GridSearchCV
        
    Returns:
    --------
    optimal_frac : float
        The optimal fraction value
    results_df : pl.DataFrame
        DataFrame with processed results
    """
    # Convert to DataFrame for easier handling
    results_df = pl.DataFrame({
        'frac': cv_results['param_frac'],
        'mean_test_score': -cv_results['mean_test_score'],  # Convert back to MSE from neg_MSE
        'std_test_score': cv_results['std_test_score'],
        'mean_train_score': -cv_results['mean_train_score'],  # Convert back to MSE
        'std_train_score': cv_results['std_train_score'],
        'rank_test_score': cv_results['rank_test_score']
    })
    
    # Get optimal fraction
    optimal_frac = results_df.filter(pl.col('rank_test_score') == 1)['frac'][0]
    
    # Sort by fraction for plotting
    results_df = results_df.sort('frac')
    
    return optimal_frac, results_df

def plot_lowess_cv_results(results_df: pl.DataFrame, optimal_frac: float) -> None:
    """Plot LOWESS cross-validation results.
    
    Parameters:
    -----------
    results_df : pl.DataFrame
        DataFrame with processed results
    optimal_frac : float
        The optimal fraction value
    """
    plt.figure(figsize=(10, 6))
    
    # Extract values
    fracs = results_df['frac'].to_numpy()
    test_scores = results_df['mean_test_score'].to_numpy()
    test_stds = results_df['std_test_score'].to_numpy()
    train_scores = results_df['mean_train_score'].to_numpy()
    train_stds = results_df['std_train_score'].to_numpy()
    
    # Plot training scores
    plt.plot(fracs, train_scores, 'o-', label='Training score', color='blue', alpha=0.7)
    plt.fill_between(fracs, 
                     train_scores - train_stds,
                     train_scores + train_stds,
                     alpha=0.1, color='blue')
    
    # Plot test scores
    plt.plot(fracs, test_scores, 'o-', label='Cross-validation score', color='red', alpha=0.7)
    plt.fill_between(fracs, 
                     test_scores - test_stds,
                     test_scores + test_stds,
                     alpha=0.1, color='red')
    
    # Mark optimal value
    optimal_score = results_df.filter(pl.col('frac') == optimal_frac)['mean_test_score'][0]
    plt.scatter([optimal_frac], [optimal_score], color='green', s=100, zorder=5,
                label=f'Optimal frac={optimal_frac:.3f}')
    
    plt.xlabel('LOWESS fraction')
    plt.ylabel('Mean Squared Error')
    plt.title('LOWESS Parameter Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optional: log scale if scores vary widely
    if np.max(test_scores) / np.min(test_scores) > 10:
        plt.yscale('log')
    
    plt.show()


def bootstrap_lowess(
    x: np.ndarray,
    y: np.ndarray,
    blocks_df: GenomicBlocks,
    frac: float = 0.5,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute LOWESS with bootstrap prediction intervals.
    
    Parameters:
    -----------
    x : np.ndarray
        x values
    y : np.ndarray
        y values
    blocks_df : GenomicBlocks
        GenomicBlocks object containing block structure
    frac : float
        LOWESS fraction parameter
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : Optional[int]
        Random seed
    """
    # Sort x and y for the main fit
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Main LOWESS fit
    y_fit = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)
    
    # Create blocks for bootstrapping
    df_blocks = blocks_df.create_blocks()
    unique_blocks = df_blocks['block_identifier'].unique().to_list()
    n_blocks = len(unique_blocks)
    
    # Create index mapping for blocks
    block_indices = {
        block: np.where(df_blocks['block_identifier'].to_numpy() == block)[0]
        for block in unique_blocks
    }
    
    # Initialize bootstrap storage
    bootstrap_fits = np.zeros((n_bootstrap, len(x)))
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Perform bootstrap iterations
    for i in range(n_bootstrap):
        # Sample blocks with replacement
        sampled_blocks = np.random.choice(unique_blocks, size=n_blocks, replace=True)
        
        # Create bootstrap sample indices
        bootstrap_indices = np.concatenate([block_indices[block] for block in sampled_blocks])
        
        x_boot = x[bootstrap_indices]
        y_boot = y[bootstrap_indices]
        
        # Sort bootstrap sample
        sort_idx_boot = np.argsort(x_boot)
        x_boot_sorted = x_boot[sort_idx_boot]
        y_boot_sorted = y_boot[sort_idx_boot]
        
        # Fit LOWESS to bootstrap sample
        y_boot_fit = lowess(y_boot_sorted, x_boot_sorted, frac=frac, xvals=x_sorted)
        bootstrap_fits[i] = y_boot_fit
    
    # Calculate prediction intervals
    lower = np.percentile(bootstrap_fits, 2.5, axis=0)
    upper = np.percentile(bootstrap_fits, 97.5, axis=0)
    
    return x_sorted, y_fit, lower, upper

