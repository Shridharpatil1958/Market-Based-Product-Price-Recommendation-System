import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Machine learning model training and evaluation framework.
    Trains multiple models for product price prediction and compares performance.
    """

    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.performance_metrics = {}
        self.best_model = None

    def prepare_data(self, df: pd.DataFrame, target_col: str,
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and target variable for model training.

        Why: Proper data preparation prevents data leakage and ensures fair evaluation.

        Args:
            df: Complete DataFrame with features
            target_col: Target column name (e.g., 'price')
            test_size: Train-test split ratio
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("DATA PREPARATION FOR TRAINING")
        print("=" * 60)

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Convert categorical columns to numeric
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_numeric = X.copy()

        for col in categorical_cols:
            X_numeric[col] = pd.factorize(X[col])[0]

        print(f"
✓ Features prepared")
        print(f"  Total samples: {len(X_numeric)}")
        print(f"  Feature columns: {X_numeric.shape[1]}")
        print(f"  Categorical columns converted: {len(categorical_cols)}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=random_state
        )

        print(f"
✓ Train-test split completed")
        print(f"  Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"  Testing samples: {len(X_test)} ({test_size*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train multiple regression models for comparison.

        Why: Ensemble of models helps identify best performer.
        Models trained:
        - Linear Regression: Fast, interpretable baseline
        - Random Forest: Captures non-linear patterns
        - Gradient Boosting: Sequential tree learning for best performance

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Dictionary of trained models
        """
        print("
" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        # Model 1: Linear Regression
        print("
1. Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.trained_models['Linear Regression'] = lr_model
        print(f"  ✓ Completed")

        # Model 2: Random Forest
        print("
2. Training Random Forest (n_estimators=100)...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        self.trained_models['Random Forest'] = rf_model
        print(f"  ✓ Completed")

        # Model 3: Gradient Boosting
        print("
3. Training Gradient Boosting (n_estimators=100)...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        self.trained_models['Gradient Boosting'] = gb_model
        print(f"  ✓ Completed")

        return self.trained_models

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate all trained models on test set.

        Why: Comprehensive evaluation identifies best performer.
        Metrics:
        - MAE: Mean Absolute Error (interpretable dollar amount)
        - RMSE: Root Mean Squared Error (penalizes large errors)
        - R² Score: Coefficient of determination (0-1 scale)

        Args:
            X_test: Testing features
            y_test: Testing target values

        Returns:
            Dictionary with evaluation metrics for each model
        """
        print("
" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)

        results = {}

        for model_name, model in self.trained_models.items():
            print(f"
{model_name}:")
            print("-" * 60)

            # Predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'predictions': y_pred
            }

            print(f"  MAE:  ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  R²:   {r2:.4f}")

            # Determine best model
            if self.best_model is None or r2 > max(
                [v['R2'] for k, v in results.items() if k != model_name]
            ):
                self.best_model = model_name

        self.performance_metrics = results
        return results

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series,
                      cv_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation for robust performance estimates.

        Why: Cross-validation provides better generalization estimates than single train-test split.

        Args:
            X_train: Training features
            y_train: Training target values
            cv_folds: Number of folds for cross-validation

        Returns:
            Dictionary with cross-validation results
        """
        print("
" + "=" * 60)
        print(f"CROSS-VALIDATION ({cv_folds}-FOLD)")
        print("=" * 60)

        cv_results = {}

        for model_name, model in self.trained_models.items():
            print(f"
{model_name}:")

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                       scoring='r2')

            cv_results[model_name] = {
                'mean_r2': cv_scores.mean(),
                'std_r2': cv_scores.std(),
                'fold_scores': cv_scores
            }

            print(f"  Mean R²:  {cv_scores.mean():.4f}")
            print(f"  Std R²:   {cv_scores.std():.4f}")
            print(f"  Folds:    {list(np.round(cv_scores, 4))}")

        return cv_results

    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> Dict:
        """
        Extract feature importance from tree-based models.

        Why: Identifies which features drive predictions.

        Args:
            model_name: Name of trained model
            feature_names: List of feature column names

        Returns:
            Dictionary with feature importances sorted by importance
        """
        print("
" + "=" * 60)
        print(f"FEATURE IMPORTANCE - {model_name}")
        print("=" * 60)

        model = self.trained_models[model_name]

        # Only tree-based models have feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"
⚠ {model_name} does not have feature_importances_")
            print(f"  Linear models use coefficients instead")
            return {}

        importances = model.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importances))

        # Sort by importance
        sorted_importance = dict(sorted(feature_importance_dict.items(),
                                       key=lambda x: x[1], reverse=True))

        print(f"
Top 10 Most Important Features:")
        print("-" * 60)

        for idx, (feature, importance) in enumerate(list(sorted_importance.items())[:10], 1):
            importance_bar = "█" * int(importance * 100)
            print(f"  {idx:2d}. {feature:25s} {importance:7.4f} {importance_bar}")

        return sorted_importance

    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk using pickle.

        Why: Persist model for later use without retraining.

        Args:
            model_name: Name of model to save
            filepath: Path where model will be saved
        """
        if model_name not in self.trained_models:
            print(f"Error: Model '{model_name}' not found")
            return

        model = self.trained_models[model_name]

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        print(f"
✓ Model saved: {filepath}")

    def load_model(self, filepath: str) -> object:
        """
        Load previously trained model from disk.

        Why: Reuse trained models without retraining.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model object
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"✓ Model loaded: {filepath}")
        return model

    def generate_report(self) -> Dict:
        """
        Generate comprehensive training report.

        Args:
            None

        Returns:
            Dictionary with complete training summary
        """
        report = {
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model,
            'performance_metrics': self.performance_metrics,
            'models_trained': list(self.trained_models.keys())
        }

        print("
" + "=" * 60)
        print("TRAINING SUMMARY REPORT")
        print("=" * 60)
        print(f"
Models trained: {report['total_models_trained']}")
        print(f"Best model: {report['best_model']}")
        print(f"Available models: {', '.join(report['models_trained'])}")

        return report
