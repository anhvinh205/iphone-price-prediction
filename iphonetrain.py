import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Config
DATA_PATH = "./data/iphone_release.csv"
LABEL_COL = "current_price(LKR)"
DROP_COLS = ["IMEI_number", "Release_Date"]

MODEL_DIR = "./models"
RESULTS_DIR = "./results"
TEST_SIZE = 0.3
RANDOM_STATE = 1


def load_dataset(path: str, label_col: str = LABEL_COL, drop_cols: list = DROP_COLS):
    """Load raw dataset and drop columns that should not be used for training."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    if drop_cols:
        print(f"[INFO] Dropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols, errors="ignore")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    X = df.drop(columns=[label_col])
    y = df[label_col].values

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(str)

    print(f"[INFO] Dataset after cleanup contains {X.shape[1]} features")
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    feature_names = numeric_features + categorical_features
    return preprocessor, feature_names


def evaluate_model(name: str, pipeline: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray):
    y_pred = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\n{'─'*45}")
    print(f"  Model : {name}")
    print(f"  MAE   : {mae:,.2f}")
    print(f"  MSE   : {mse:,.2f}")
    print(f"  R²    : {r2:.4f}")
    print(f"{'─'*45}")

    return {"Model": name, "MAE": round(mae, 2), "MSE": round(mse, 2), "R2": round(r2, 4)}


def plot_results(results: list, save_path: str):
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#4C9BE8", "#E8734C", "#4CAF82"]
    metrics = ["MAE", "MSE", "R2"]
    titles = [
        "MAE (thấp hơn = tốt hơn)",
        "MSE (thấp hơn = tốt hơn)",
        "R² (cao hơn = tốt hơn)",
    ]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        bars = ax.bar(df["Model"], df[metric], color=color, edgecolor="white")
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=9)
        ax.bar_label(bars, fmt="%.2f" if metric != "MSE" else "%.0f", padding=3, fontsize=8)
        ax.set_ylabel(metric)

    fig.suptitle("iPhone Resale Price — Model Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Plot saved → {save_path}")


def save_pipeline(pipeline: Pipeline, model_name: str):
    path = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_').lower()}.joblib")
    joblib.dump(pipeline, path)
    print(f"[INFO] Saved pipeline → {path}")


def train_models(X_train, X_val, y_train, y_val, preprocessor: Pipeline, feature_names: list):
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        print(f"\n[TRAIN] {name} ...")
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
        pipeline.fit(X_train, y_train)

        results.append(evaluate_model(name, pipeline, X_val, y_val))
        save_pipeline(pipeline, name)

    return results


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, y = load_dataset(DATA_PATH)
    preprocessor, feature_names = build_preprocessor(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"[INFO] Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    results = train_models(X_train, X_val, y_train, y_val, preprocessor, feature_names)

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Metrics:\n{results_df.to_string(index=False)}")

    plot_results(results, os.path.join(RESULTS_DIR, "comparison.png"))

    best = results_df.loc[results_df["R2"].idxmax(), "Model"]
    print(f"\n✅ Best model (R²): {best} (R²={results_df['R2'].max():.4f})")


if __name__ == "__main__":
    main()
 