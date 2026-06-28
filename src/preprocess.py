import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess(df):
    # filtering
    df = df.dropna(subset=["target"])
    df = df[df["on_thyroxine"] == "t"]

    features = [
        "TSH", "T3", "TT4", "T4U", "FTI",
        "age", "sex",
        "thyroid_surgery", "I131_treatment",
        "hypopituitary", "goitre", "pregnant"
    ]
    df = df[features]

    numeric_cols = ["TSH", "T3", "TT4", "T4U", "FTI", "age"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    binary_map = {"t": 1, "f": 0}
    for col in ["thyroid_surgery", "I131_treatment", "hypopituitary", "goitre", "pregnant"]:
        df[col] = df[col].map(binary_map)
    df["sex"] = df["sex"].map({"M": 0, "F": 1})

    # missing vals heatmap
    missing_count = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100

    numeric_features = ["TSH", "T3", "TT4", "T4U", "FTI", "age"]
    categorical_features = ["sex", "thyroid_surgery", "I131_treatment", "hypopituitary", "goitre", "pregnant"]

    num_pct = missing_pct[numeric_features]
    num_count = missing_count[numeric_features]
    cat_pct = missing_pct[categorical_features]
    cat_count = missing_count[categorical_features]

    num_order = num_pct.sort_values(ascending=False).index
    cat_order = cat_pct.sort_values(ascending=False).index

    num_color = np.vstack([num_pct[num_order].values, num_pct[num_order].values])
    num_annot = np.array([
        [f"{int(c)}" for c in num_count[num_order].values],
        [f"{p:.2f}%" for p in num_pct[num_order].values]
    ])

    cat_color = np.vstack([cat_pct[cat_order].values, cat_pct[cat_order].values])
    cat_annot = np.array([
        [f"{int(c)}" for c in cat_count[cat_order].values],
        [f"{p:.2f}%" for p in cat_pct[cat_order].values]
    ])

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    sns.heatmap(
        num_color,
        annot=num_annot,
        fmt="",
        cmap="Reds",
        cbar=True,
        cbar_kws={"label": "Missing %"},
        annot_kws={"size": 9},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        xticklabels=num_order,
        yticklabels=["Count", "Missing %"],
        ax=axes[0]
    )
    axes[0].set_title("Numeric Features")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(
        cat_color,
        annot=cat_annot,
        fmt="",
        cmap="Reds",
        cbar=True,
        cbar_kws={"label": "Missing %"},
        annot_kws={"size": 9},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        xticklabels=cat_order,
        yticklabels=["Count", "Missing %"],
        ax=axes[1]
    )
    axes[1].set_title("Categorical Features")
    axes[1].set_xlabel("Features")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.suptitle("Feature-wise Missing Value Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig("missingness_feature_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # imputation + log + standardize
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = ["sex", "thyroid_surgery", "I131_treatment", "hypopituitary", "goitre", "pregnant"]
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in ["TSH", "T3", "TT4", "T4U", "FTI"]:
        df[col] = np.log1p(df[col])

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df