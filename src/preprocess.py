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
    plt.figure(figsize=(10, 3))

    missing_count = df.isnull().sum()
    missing_pct = df.isnull().mean() * 100

    order = missing_pct.sort_values(ascending=False).index
    missing_count = missing_count[order]
    missing_pct = missing_pct[order]

    color_matrix = np.vstack([missing_pct.values, missing_pct.values])

    annot_matrix = np.array([
        [f"{int(c)}" for c in missing_count.values],
        [f"{p:.2f}%" for p in missing_pct.values]
    ])

    sns.heatmap(
        color_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="Reds",
        cbar=True,
        cbar_kws={"label": "Missing %"},
        annot_kws={"size": 9},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        xticklabels=order,
        yticklabels=["Count", "Missing %"]
    )

    plt.title("Feature-wise Missing Value Summary")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

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