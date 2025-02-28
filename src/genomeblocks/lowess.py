from typing import Callable, Optional, List, Union, Dict, Tuple
import warnings 
import polars as pl
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass
from sklearn.model_selection._split import _BaseKFold
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from genomeblocks import GenomicBlocks


class BlockCV(_BaseKFold):
    """Cross-validation iterator that respects genomic block structure."""
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.blocks: Optional[GenomicBlocks] = None
        self.block_assignments: Optional[dict[str, int]] = None
    
    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test sets."""
        if self.blocks is None:
            raise ValueError("Must call set_blocks before splitting")
            
        # Lazily create block assignments if not already done
        if self.block_assignments is None:
            self._assign_blocks_to_folds()
            
        blocks_df = self.blocks.create_blocks()
        point_folds = np.array([
            self.block_assignments[block] 
            for block in blocks_df['block_identifier'].to_list()
        ])
        
        for fold in range(self.n_splits):
            test_mask = point_folds == fold
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]
    
    def set_blocks(self, blocks: GenomicBlocks) -> 'BlockCV':
        """Set the GenomicBlocks object containing block information."""
        self.blocks = blocks
        self.block_assignments = None
        return self
        
    def _assign_blocks_to_folds(self) -> None:
        """Randomly assign blocks to folds."""
        if self.blocks is None:
            raise ValueError("Blocks not set")
            
        blocks_df = self.blocks.create_blocks()
        unique_blocks = blocks_df['block_identifier'].unique().to_list()
        rng = np.random.RandomState(self.random_state)
        
        self.block_assignments = {
            block: fold for block, fold in zip(
                unique_blocks,
                rng.randint(0, self.n_splits, size=len(unique_blocks))
            )
        }


class LOWESSRegressor(BaseEstimator, RegressorMixin):
    """LOWESS regression compatible with scikit-learn API."""
    
    def __init__(self, frac: float = 0.5):
        self.frac = frac
    
    def fit(self, X, y):
        self.X_train_ = X.ravel() if X.ndim > 1 else X
        self.y_train_ = y
        return self
    
    def predict(self, X):
        X = X.ravel() if X.ndim > 1 else X
        return lowess(self.y_train_, self.X_train_, xvals=X, frac=self.frac)


@dataclass
class BlockLOWESS:
    """LOWESS analysis with block-aware cross-validation and bootstrapping."""
    
    blocks: GenomicBlocks
    x_col: str
    y_col: str
       
    def bootstrap_fit(
        self,
        frac: float = 0.5,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LOWESS with bootstrap prediction intervals."""
        x = self.blocks.df[self.x_col].to_numpy()
        y = self.blocks.df[self.y_col].to_numpy()
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        # Main fit
        y_fit = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)
        
        # Setup for bootstrapping
        blocks_df = self.blocks.create_blocks()
        unique_blocks = blocks_df['block_identifier'].unique().to_list()
        block_indices = {
            block: np.where(blocks_df['block_identifier'].to_numpy() == block)[0]
            for block in unique_blocks
        }
        
        # Bootstrap
        if random_state is not None:
            np.random.seed(random_state)
            
        bootstrap_fits = np.zeros((n_bootstrap, len(x)))
        for i in tqdm(range(n_bootstrap)):
            sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
            bootstrap_indices = np.concatenate([block_indices[block] for block in sampled_blocks])
            
            x_boot = x[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            sort_idx_boot = np.argsort(x_boot)
            x_boot_sorted = x_boot[sort_idx_boot]
            y_boot_sorted = y_boot[sort_idx_boot]
            
            bootstrap_fits[i] = lowess(y_boot_sorted, x_boot_sorted, frac=frac, xvals=x_sorted)
        
        lower = np.percentile(bootstrap_fits, 2.5, axis=0)
        upper = np.percentile(bootstrap_fits, 97.5, axis=0)
        
        return x_sorted, y_fit, lower, upper
    
     
    def optimize_parameters(
        self,
        param_grid: Optional[dict] = None,
        n_splits: int = 10,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        min_diff_threshold: float = 1e-6  # Minimum meaningful difference in MSE
    ) -> tuple[LOWESSRegressor, dict, pl.DataFrame]:
        """Find optimal LOWESS parameters using block-aware cross-validation.
        
        Parameters:
        -----------
        param_grid : Optional[dict]
            Grid of parameters to search. If None, uses custom grid focused on lower fractions
        n_splits : int
            Number of CV folds
        random_state : Optional[int]
            Random seed
        n_jobs : int
            Number of parallel jobs
        min_diff_threshold : float
            Minimum difference in MSE to consider meaningful
        """
        if param_grid is None:
            # Focus grid on lower fractions with finer resolution
            param_grid = {
                'frac': np.concatenate([
                    np.linspace(0.1, 0.3, 10),  # Fine grid in diagnostic range
                    np.linspace(0.4, 0.9, 5)     # Coarse grid for higher values
                ])
            }
            
        cv = BlockCV(n_splits=n_splits, random_state=random_state)
        cv.set_blocks(self.blocks)
        
        X = self.blocks.df[self.x_col].to_numpy().reshape(-1, 1)
        y = self.blocks.df[self.y_col].to_numpy()
        
        grid_search = GridSearchCV(
            estimator=LOWESSRegressor(),
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        grid_search.fit(X, y)
        
        # Process results
        results_df = pl.DataFrame({
            'frac': grid_search.cv_results_['param_frac'],
            'mean_test_score': -grid_search.cv_results_['mean_test_score'],
            'std_test_score': grid_search.cv_results_['std_test_score'],
            'mean_train_score': -grid_search.cv_results_['mean_train_score'],
            'std_train_score': grid_search.cv_results_['std_train_score'],
            'rank_test_score': grid_search.cv_results_['rank_test_score']
        }).sort('frac')
        
        # Analyze results for potential issues
        best_score = results_df.filter(pl.col('rank_test_score') == 1)
        best_frac = best_score['frac'][0]
        
        # Check if scores are too similar
        score_range = (results_df['mean_test_score'].max() - 
                      results_df['mean_test_score'].min())
        
        if score_range < min_diff_threshold:
            warnings.warn("Very small differences in MSE across fractions")

        if best_frac > 0.5:
            warnings.warn(f"Warning: Selected fraction {best_frac:.3f} > 0.5 may be too smooth\n"
                           "Consider using a smaller fraction for diagnostic purposes")

        return grid_search.best_estimator_, grid_search.cv_results_, results_df
    
    def plot_optimization_results(
        self, 
        results_df: pl.DataFrame,
        show_warning_threshold: bool = True
    ) -> None:
        """Plot cross-validation results with additional diagnostics."""
        optimal_frac = results_df.filter(pl.col('rank_test_score') == 1)['frac'][0]
        optimal_score = results_df.filter(pl.col('frac') == optimal_frac)['mean_test_score'][0]
        
        plt.figure(figsize=(12, 6))
        
        # Extract values
        fracs = results_df['frac'].to_numpy()
        test_scores = results_df['mean_test_score'].to_numpy()
        test_stds = results_df['std_test_score'].to_numpy()
        train_scores = results_df['mean_train_score'].to_numpy()
        train_stds = results_df['std_train_score'].to_numpy()
        
        # Plot scores
        plt.plot(fracs, train_scores, 'o-', label='Training score', color='blue', alpha=0.7)
        plt.fill_between(fracs, train_scores - train_stds, train_scores + train_stds,
                        alpha=0.1, color='blue')
        
        plt.plot(fracs, test_scores, 'o-', label='Cross-validation score', color='red', alpha=0.7)
        plt.fill_between(fracs, test_scores - test_stds, test_scores + test_stds,
                        alpha=0.1, color='red')
        
        # Mark optimal and recommended diagnostic fraction
        plt.scatter([optimal_frac], [optimal_score], color='green', s=100, zorder=5,
                   label=f'Selected frac={optimal_frac:.3f}')
        
        if show_warning_threshold:
            plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5,
                       label='Warning threshold')
        
        plt.xlabel('LOWESS fraction')
        plt.ylabel('Mean Squared Error')
        plt.title('LOWESS Parameter Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if np.max(test_scores) / np.min(test_scores) > 10:
            plt.yscale('log')
        
        plt.show()

    def plot_fit(
        self,
        x: np.ndarray,
        y_fit: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        hide_scatter: bool = False,
        color_lowess='blue',
        title: Optional[str] = None,
        label_point: Optional[str] = None,
        label_ci: Optional[str] = None,
        label_lowess: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figax=None,
    ):
        """Plot LOWESS fit with confidence intervals and data points.
        """

        if figax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figax

        if not hide_scatter:
            # Plot raw data
            ax.scatter(
                self.blocks.df[self.x_col].to_numpy(), 
                self.blocks.df[self.y_col].to_numpy(),
                alpha=0.3, 
                color='gray',
                s=30,
                label=label_point,
            )
        
        # # Plot confidence interval
        if lower is not None:
            assert upper is not None, "if lower is set, upper must be set."
            ax.fill_between(
                x, lower, upper,
                alpha=0.2,
                color=color_lowess,
                label=label_ci,
                linewidth=0,
            )
         
        # Plot LOWESS fit
        ax.plot(
            x, y_fit,
            color=color_lowess,
            linewidth=2,
            label=label_lowess,
        )
        
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        # ax.grid(True, alpha=0.3)

        return fig, ax
