import colorsys
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm, ListedColormap
from load_data import load_data
from preprocess import preprocess
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import gower
import seaborn as sns
import numpy as np

df_raw = load_data()  
df = preprocess(df_raw)

categorical_cols = ["sex", "thyroid_surgery", "I131_treatment", "hypopituitary", "goitre", "pregnant"]
cat_idx = [df.columns.get_loc(col) for col in categorical_cols]

random_seed = rand.randint(1, 1000)
print(f"Final model random state: {random_seed}")

# elbow method
costs = []
K_range = range(2, 10)
for k in K_range:
    print(f"Running k={k}...")
    kproto = KPrototypes(
    n_clusters=k,
    init='Cao',
    random_state=random_seed
    # random_seed used in paper = 42
)
    clusters = kproto.fit_predict(df, categorical=cat_idx)
    norm_cost = kproto.cost_ / len(df)
    costs.append(norm_cost)
    print(f"k={k}, normalized_cost={norm_cost}")

print("\nNormalized cost differences:")
for i in range(1, len(costs)):
    delta = costs[i-1] - costs[i]
    print(f"k={list(K_range)[i-1]}→{list(K_range)[i]}: Δ={delta:.4f}")

plt.plot(K_range, costs, marker='o')
optimal_idx = list(K_range).index(5)
plt.annotate(
    'k=5 (optimal)',
    xy=(5, costs[optimal_idx]),
    xytext=(6, costs[optimal_idx] + 0.3),
    arrowprops=dict(arrowstyle='->', color='red'),
    color='red',
    fontsize=9
)
plt.scatter([5], [costs[optimal_idx]], color='red', zorder=5)
plt.xlabel('Number of clusters k')
plt.ylabel('Normalized Cost')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.grid(True)
plt.savefig("elbow_plot.png", dpi=300, bbox_inches='tight')
plt.show()

pd.DataFrame({
    "k": list(K_range),
    "cost": costs
}).to_csv("elbow_results.csv", index=False)

optimal_k = 5 # chosen from elbow

# ari
labels_list = []
# seeds used in paper = [729, 725, 635, 945, 521]
seeds = rand.sample(range(1,1000),5)
print(f"Seeds used: {seeds}")

for seed in seeds:
    kproto = KPrototypes(n_clusters=optimal_k, init='Cao', random_state=seed)
    labels = kproto.fit_predict(df, categorical=cat_idx)
    labels_list.append(labels)

ari_results = []
for i in range(len(labels_list)):
    for j in range(i+1, len(labels_list)):
        ari = adjusted_rand_score(labels_list[i], labels_list[j])
        print(f"ARI between run {i} and {j}: {ari:.4f}")
        ari_results.append({"run_i": i, "run_j": j, "ari": round(ari, 4)})

ari_values = [entry['ari'] for entry in ari_results]
print(f"\nARI range: {min(ari_values):.4f} to {max(ari_values):.4f}")
print(f"Mean ARI: {sum(ari_values)/len(ari_values):.4f}")

pd.DataFrame(ari_results).to_csv("ari_stability_results.csv", index=False)

n_runs = len(labels_list)
ari_matrix = np.zeros((n_runs, n_runs))
for entry in ari_results:
    i, j, ari = entry['run_i'], entry['run_j'], entry['ari']
    ari_matrix[i][j] = ari
    ari_matrix[j][i] = ari
np.fill_diagonal(ari_matrix, 1.0)

plt.figure(figsize=(6, 5))
sns.heatmap(
    ari_matrix,
    annot=True,
    fmt='.4f',
    cmap='YlGn',
    vmin=0.9,
    vmax=1.0,
    xticklabels=[f'Run {i}' for i in range(n_runs)],
    yticklabels=[f'Run {i}' for i in range(n_runs)],
    cbar_kws={'label': 'Adjusted Rand Index'}
)
plt.title('Cluster Stability Across Random Initializations (ARI)')
plt.tight_layout()
plt.savefig("ari_stability_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# final clustering
kproto = KPrototypes(
    n_clusters=optimal_k,
    init='Cao',
    random_state=random_seed
    # random_seed used in paper = 42
)

clusters = kproto.fit_predict(df, categorical=cat_idx)

df['cluster'] = clusters
print("\nCluster sizes:")
print(df['cluster'].value_counts())

cluster_counts = df['cluster'].value_counts().sort_index()

plt.figure(figsize=(7, 4))
bars = plt.bar(
    [f'Cluster {i}' for i in cluster_counts.index],
    cluster_counts.values,
    color='steelblue',
    edgecolor='white'
)

for bar, val in zip(bars, cluster_counts.values):
    padding = max(3, val*0.02)
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + padding,
        str(val),
        ha='center',
        va='bottom',
        fontsize=10
    )

plt.xlabel('Cluster')
plt.ylabel('Number of patients')
plt.title('Patient Count by Cluster')
plt.ylim(0, cluster_counts.values.max() * 1.15)
plt.tight_layout()
plt.savefig("cluster_sizes.png", dpi=300, bbox_inches='tight')
plt.show()

# silhouette score
print("Computing Gower distance matrix...")
cat_mask = [col in categorical_cols for col in df.drop(columns='cluster').columns]
gower_dist = gower.gower_matrix(df.drop(columns='cluster'), cat_features=cat_mask)

sil_score = silhouette_score(gower_dist, df['cluster'], metric='precomputed')
print(f"Gower-based Silhouette Score: {sil_score:.4f}")

sample_scores = silhouette_samples(gower_dist, df['cluster'], metric='precomputed')
df['silhouette'] = sample_scores
sil_by_cluster = df.groupby('cluster')['silhouette'].mean()
print("Per-cluster silhouette scores:")
print(sil_by_cluster)

fig, ax = plt.subplots(figsize=(10, 7))
y_lower = 10

for i in sorted(df['cluster'].unique()):
    cluster_sil = df[df['cluster'] == i]['silhouette'].values.copy()
    cluster_sil.sort()
    size = len(cluster_sil)
    y_upper = y_lower + size
    cluster_mean = sil_by_cluster[i]

    bars = ax.barh(range(y_lower, y_upper), cluster_sil, height=1.0,
                   alpha=0.85, label=f'Cluster {i}')
    color = bars[0].get_facecolor()

    ax.text(-0.30, (y_lower + y_upper) / 2,
            f'Cluster {i}  (n={size})',
            fontsize=10, va='center', ha='left', color=color, fontweight='bold')

    ax.text(cluster_sil.max() + 0.01, (y_lower + y_upper) / 2,
            f'mean={cluster_mean:.3f}',
            fontsize=10, va='center', ha='left', color=color)

    y_lower = y_upper + 15

ax.axvline(x=sil_score, color='black', linestyle='--', linewidth=1,
           label=f'Overall: {sil_score:.4f}')
ax.axvline(x=0, color='grey', linewidth=0.5)

ax.set_xlabel('Silhouette coefficient', fontsize=11)
ax.set_title('Silhouette Plot by Cluster (Gower Distance)', fontsize=13, pad=12)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend(loc='lower right', frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=300, bbox_inches='tight')
plt.show()

pd.DataFrame({
    "metric": ["gower_silhouette"] + [f"cluster_{i}_silhouette" for i in sil_by_cluster.index],
    "score": [round(sil_score, 4)] + sil_by_cluster.round(4).tolist()
}).to_csv("silhouette_score.csv", index=False)

# cluster summary + visuals
numeric_cols = ["TSH","T3","TT4","T4U","FTI","age"]
summary_stats = df.groupby('cluster')[numeric_cols].agg(['mean', 'std']).round(3)
print("\nNumeric feature means and standard deviation by cluster")
print(summary_stats)
summary_stats.to_csv("cluster_summary_stats.csv", index=False)

numeric_summary = summary_stats.xs('mean', axis=1, level=1)

categorical_summary = df.groupby('cluster')[categorical_cols].mean()
print("\nCategorical feature distributions by cluster (fraction of 1s):")
print(categorical_summary)

print("\nNumeric features (population z-scores):")
print(numeric_summary.round(3))
print("\nCategorical features (proportions):")
print(categorical_summary.round(3))

fig, axes = plt.subplots(
    2, 1,
    figsize=(8, 10),
    gridspec_kw={'height_ratios': [1, 1]}
)

sns.heatmap(
    numeric_summary,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    center=0,
    linewidths=0.5,
    ax=axes[0],
    cbar_kws={'label': 'Z-score'}
)
axes[0].set_title('Numeric Features (Population Z-scores)')
axes[0].set_ylabel('Cluster')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(
    categorical_summary,
    annot=True,
    fmt='.3f',
    cmap='Blues',
    vmin=0,
    vmax=1,
    linewidths=0.5,
    ax=axes[1],
    cbar_kws={'label': 'Proportion'}
)
axes[1].set_title('Categorical Features (Proportions)')
axes[1].set_ylabel('')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Cluster Feature Profiles', fontsize=13)
plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[numeric_cols])

print("\nExplained variance ratio (PC1, PC2):")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))

def adjust_saturation(color, factor=1.6):
    r, g, b, a = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0.0, s * factor)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, a)

colors = plt.cm.get_cmap('Set3')
selected_colors = [adjust_saturation(colors(i)) for i in [4, 5, 0, 3, 2]]
custom_cmap = ListedColormap(selected_colors)

scatter = plt.scatter(
    df_pca[:,0],
    df_pca[:,1],
    c=df['cluster'],
    cmap=custom_cmap,
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cluster Visualization (PCA of Numeric Features)')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.savefig("thyroid_clusters_pca.png", dpi=300, bbox_inches='tight')
plt.show()

df.to_csv("thyroid_clusters.csv", index=False)