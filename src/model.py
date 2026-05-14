import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # ── Grid search ──────────────────────────────────────────
    param_grid = {
        "n_estimators": [400],
        
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }

    base_model = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
        verbosity=0
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")

    # ── Retrain best model ───────────────────────────────────
    best_model = XGBRegressor(
        **grid_search.best_params_,
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=20
    )

    best_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluation ───────────────────────────────────────────
    predictions = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE on test set: {mae:.3f} seconds")

    # ── Feature importance ───────────────────────────────────
    importance = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    for feature, score in importance.items():
        print(f"  {feature:<20} {score:.4f}")

    print("\nModel training completed successfully ✅")

    return best_model