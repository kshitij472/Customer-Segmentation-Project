Customer Segmentation Project

Project Overview

This project segments e-commerce customers based on their income and purchasing behavior using Python and K-Means clustering.

The goal is to identify groups of customers with similar characteristics so that businesses can better understand customer behavior and design targeted marketing strategies.

Objectives

Explore and understand customer data

Check data quality, missing values, and duplicates

Perform exploratory data analysis (EDA)

Select important behavioral and demographic features

Scale the selected features

Apply K-Means clustering

Determine a suitable number of clusters using the Elbow Method and Silhouette Score

Profile the resulting customer segments

Visualize the customer segments

Export the clustering results for further analysis

Dataset

The project uses a sample e-commerce customer dataset containing 100 customer records and 40 columns.

Important features used for segmentation:

annual_income

spending_score

purchase_frequency

avg_order_value

Note: The dataset is a sample/simulated dataset and should not be interpreted as real customer data.

Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

K-Means Clustering

Project Workflow

1. Data Loading

The dataset is loaded using Pandas.

2. Data Quality Check

The project checks:

Dataset shape

Missing values

Duplicate rows

Duplicate customer IDs

Descriptive statistics

3. Exploratory Data Analysis

The project visualizes:

Customer age distribution

Annual income distribution

Spending score distribution

4. Feature Selection

The following four variables are used for clustering:

Annual Income
Spending Score
Purchase Frequency
Average Order Value

5. Feature Scaling

StandardScaler is used to standardize the selected features so that variables with different numerical ranges can be compared fairly.

6. Finding the Number of Clusters

Two methods are used:

Elbow Method

Silhouette Score

The analysis selected 7 clusters for the final K-Means model based on the evaluated silhouette scores.

7. K-Means Clustering

K-Means assigns each customer to a cluster based on similarity in the selected behavioral and demographic features.

8. Customer Segment Analysis

The project calculates the average characteristics of each cluster to understand how the customer groups differ.

9. Visualization

The final segmentation is visualized using:

Income vs Spending Score scatter plot

Cluster size chart

EDA charts

Elbow curve

Silhouette score chart

Project Structure

Task 2 - Customer Segmentation/
│
├── customer_segmentation.py
├── ecomerce.csv
├── customer_segments.csv
├── cluster_summary.csv
├── figures/
│   ├── age_distribution.png
│   ├── income_distribution.png
│   ├── spending_score_distribution.png
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── customer_segments.png
│   └── cluster_sizes.png
└── README.md

How to Run

1. Install required libraries

pip install pandas numpy matplotlib seaborn scikit-learn

2. Keep the dataset and Python file in the same folder

customer_segmentation.py
ecomerce.csv

3. Run the project

python customer_segmentation.py

The script generates the clustering results and saves the output CSV files and visualizations.

Output Files

customer_segments.csv

Contains each customer along with their assigned cluster.

cluster_summary.csv

Contains the average values of the selected features for each cluster.

figures/

Contains the generated charts used for EDA and segmentation analysis.

Key Learning Outcomes

Through this project, I practiced:

Data cleaning

Exploratory Data Analysis

Feature selection

Feature scaling

Unsupervised Machine Learning

K-Means clustering

Elbow Method

Silhouette Score

Customer profiling

Data visualization

Business-oriented customer segmentation

Conclusion

The project demonstrates how customer data can be analyzed and grouped into meaningful segments using K-Means clustering. These segments can help businesses understand differences in customer income, spending behavior, purchase frequency, and average order value.

Author

Kshitij