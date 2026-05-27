import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# config
DATA_PATH = "./data/iphone_release.csv"
RESULTS_DIR = "./results"
# target/label column in the CSV
label_col = "current_price(LKR)"

def run_eda(path: str = DATA_PATH):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(path)

    print("=" * 55)
    print(f"  DATASET: {path}")
    print(f"  Shape  : {df.shape[0]} rows × {df.shape[1]} cols")
    print("=" * 55)
    print("\n--- Columns ---")
    print(df.dtypes.to_string())
    print("\n--- First 5 rows ---")
    print(df.head().to_string())
    print("\n--- Missing values ---")
    print(df.isnull().sum().to_string())
    print("\n--- Descriptive statistics ---")
    print(df.describe().to_string())
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
 
    # 1. Label distribution
    if label_col in df.columns:
        axes[0].hist(df[label_col].dropna(), bins=30, color="#4C9BE8", edgecolor="white")
        axes[0].set_title(f"Distribution of {label_col}")
        axes[0].set_xlabel(label_col)
    else:
        axes[0].text(0.5, 0.5, f"Column '{label_col}' not found", ha="center", va="center")
        axes[0].set_title("Label not found")
 
    # 2. Correlation heatmap (numeric only)
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] >= 2:
        sns.heatmap(num_df.corr(), ax=axes[1], annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5, annot_kws={"size": 8})
        axes[1].set_title("Correlation Heatmap")
    else:
        axes[1].text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
 
    fig.tight_layout()
    out = f"{RESULTS_DIR}/eda_overview.png"
    fig.savefig(out, dpi=150)
    print(f"\n[INFO] EDA plot saved → {out}")
 
    # 3. Boxplots for top categorical cols vs label
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols and label_col in df.columns:
        top_cats = cat_cols[:3]
        fig2, axes2 = plt.subplots(1, len(top_cats), figsize=(5 * len(top_cats), 4))
        if len(top_cats) == 1:
            axes2 = [axes2]
        for ax, col in zip(axes2, top_cats):
            order = df.groupby(col)[label_col].median().sort_values(ascending=False).index
            sns.boxplot(x=col, y=label_col, data=df, order=order, ax=ax, palette="pastel")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{label_col} by {col}")
            ax.set_xlabel(col)
        fig2.suptitle("")
        fig2.tight_layout()
        out2 = f"{RESULTS_DIR}/eda_categorical.png"
        fig2.savefig(out2, dpi=150)
        print(f"[INFO] Categorical plot saved → {out2}")
 
 
if __name__ == "__main__":
    run_eda()
 