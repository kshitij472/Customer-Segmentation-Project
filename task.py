# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# ============================================================

# STEP 1: Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# STEP 2: Load Dataset
# ============================================================

df = pd.read_csv("ecomerce.csv")

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# ============================================================
# STEP 3: Basic Data Inspection
# ============================================================

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== DUPLICATE CUSTOMER IDs ==========")
print(df["customer_id"].duplicated().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# ============================================================
# STEP 4: Exploratory Data Analysis
# ============================================================

# Create folder for graphs
os.makedirs("figures", exist_ok=True)

sns.set_theme(style="whitegrid")


# ----- Age Distribution -----
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("figures/age_distribution.png")
plt.show()


# ----- Annual Income Distribution -----
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="annual_income", bins=10, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("figures/income_distribution.png")
plt.show()


# ----- Spending Score Distribution -----
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="spending_score", bins=10, kde=True)
plt.title("Customer Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("figures/spending_score_distribution.png")
plt.show()


# ============================================================
# STEP 5: Select Features for Customer Segmentation
# ============================================================

features = [
    "annual_income",
    "spending_score",
    "purchase_frequency",
    "avg_order_value"
]

X = df[features].copy()

print("\n========== SELECTED FEATURES ==========")
print(X.head())

print("\nFeature Shape:", X.shape)


# ============================================================
# STEP 6: Feature Scaling
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== SCALED DATA ==========")
print(X_scaled[:5])


# ============================================================
# STEP 7: Find Suitable Number of Clusters
#        Using Elbow Method + Silhouette Score
# ============================================================

inertia = []
silhouette_scores = []

K_range = range(2, 9)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )


# ----- Elbow Plot -----
plt.figure(figsize=(8, 5))
plt.plot(
    list(K_range),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(K_range))
plt.tight_layout()
plt.savefig("figures/elbow_method.png")
plt.show()


# ----- Silhouette Score Plot -----
plt.figure(figsize=(8, 5))
plt.plot(
    list(K_range),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(K_range))
plt.tight_layout()
plt.savefig("figures/silhouette_score.png")
plt.show()


# Select K with highest silhouette score
best_k = list(K_range)[np.argmax(silhouette_scores)]

print("\n========== CLUSTER SELECTION ==========")
print("Best number of clusters:", best_k)

for k, score in zip(K_range, silhouette_scores):
    print(f"K = {k} | Silhouette Score = {score:.3f}")


# ============================================================
# STEP 8: Apply K-Means Clustering
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n========== CLUSTER ASSIGNMENTS ==========")
print(df[["customer_id", "Cluster"]].head(20))


# ============================================================
# STEP 9: Cluster Summary
# ============================================================

cluster_summary = df.groupby("Cluster")[features].mean().round(2)

print("\n========== CLUSTER PROFILE ==========")
print(cluster_summary)


# Customer count in each cluster
cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n========== CUSTOMERS PER CLUSTER ==========")
print(cluster_counts)


# ============================================================
# STEP 10: Visualize Customer Segments
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="annual_income",
    y="spending_score",
    hue="Cluster",
    size="purchase_frequency",
    sizes=(40, 300),
    palette="tab10"
)

plt.title("Customer Segmentation: Income vs Spending Score")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("figures/customer_segments.png")
plt.show()


# ============================================================
# STEP 11: Cluster Size Visualization
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster",
    hue="Cluster",
    palette="tab10",
    legend=False
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("figures/cluster_sizes.png")
plt.show()


# ============================================================
# STEP 12: Save Results
# ============================================================

df.to_csv(
    "customer_segments.csv",
    index=False
)

cluster_summary.to_csv(
    "cluster_summary.csv"
)

print("\n========== FILES CREATED ==========")
print("1. customer_segments.csv")
print("2. cluster_summary.csv")
print("3. figures/ folder")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n======================================")
print("CUSTOMER SEGMENTATION COMPLETED")
print("======================================")