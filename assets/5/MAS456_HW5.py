#!/usr/bin/env python
# coding: utf-8

# In[1]:


############### Question 2 #################



## a ##


import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

m = np.array([[0, 0.3, 0.4, 0.7],
              [0.3, 0, 0.5, 0.8],
              [0.4, 0.5, 0, 0.45],
              [0.7, 0.8, 0.45, 0]])

# Performing hierarchical clustering
c1 = sch.linkage(sch.distance.squareform(m), method='complete')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(c1)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()


# In[2]:


### b ###
c2 = sch.linkage(sch.distance.squareform(m), method='single')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(c2)
plt.title("Linkage Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()


# In[3]:


#### c ####

# Forming two clusters
clusters = sch.fcluster(c1, t=2, criterion='maxclust')

# Identifying observations in each cluster
cluster_1 = np.where(clusters == 1)[0]
cluster_2 = np.where(clusters == 2)[0]

cluster_1, cluster_2


# In[4]:


#### d ####

# Forming two clusters
clusters = sch.fcluster(c2, t=2, criterion='maxclust')

# Identifying observations in each cluster
cluster_1 = np.where(clusters == 1)[0]
cluster_2 = np.where(clusters == 2)[0]

cluster_1, cluster_2


# In[5]:


### e ###

# Assuming m is already defined as before
c1 = sch.linkage(sch.distance.squareform(m), method='complete')

# Plotting the dendrogram with reordered leaves
plt.figure(figsize=(10, 7))
sch.dendrogram(c1, labels=[1, 0, 2, 3])
plt.title("Repositioned Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()


# In[6]:


################ Problem 3 #################

### a ###

import matplotlib.pyplot as plt

# Data points
x1 = [1, 1, 0, 5, 6, 4]
x2 = [4, 3, 4, 1, 2, 0]

# Creating the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, color='blue')
plt.title("Scatter Plot of Observations")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.grid(True)
plt.show()


# In[7]:


### b ###

# Setting the seed for reproducibility
np.random.seed(42)

# Observations
observations = np.array([[1, 4], [1, 3], [0, 4], [5, 1], [6, 2], [4, 0]])

# Randomly assigning cluster labels
cluster_labels = np.random.choice([1, 2], size=len(observations), replace=True)

# Combine observations with their cluster labels
cluster_assignments = list(zip(observations, cluster_labels))
print(cluster_assignments)


# In[8]:


### c ###


# Initialize an empty list to store centroids
centroids = []

# Compute the centroid for each cluster
for i in [1, 2]:
    cluster_points = observations[cluster_labels == i]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append(centroid)

print(centroids)


# In[9]:


### d ###

# Calculate Euclidean distances and assign new cluster labels
new_cluster_labels = []
for obs in observations:
    distances = [np.sqrt(np.sum((obs - centroid)**2)) for centroid in centroids]
    new_cluster_labels.append(np.argmin(distances) + 1)  # Adding 1 because cluster labels are 1 and 2

print(new_cluster_labels)


# In[10]:


### e ###
observations = np.array([[1, 4], [1, 3], [0, 4], [5, 1], [6, 2], [4, 0]])
np.random.seed(42)
cluster_labels = np.random.choice([1, 2], size=len(observations), replace=True)

# print(centroids, cluster_labels)

# Function to compute centroids
def compute_centroids(observations, labels):
    return [np.mean(observations[labels == i], axis=0) for i in [1, 2]]

# Function to assign new cluster labels based on centroids
def assign_clusters(observations, centroids):
    new_labels = []
    for obs in observations:
        distances = [np.sqrt(np.sum((obs - centroid) ** 2)) for centroid in centroids]
#        print(distances)
        new_labels.append(np.argmin(distances) + 1)
#    print("")
    return np.array(new_labels)

# Repeating steps (c) and (d) until the cluster assignments stop changing
while True:
    centroids = compute_centroids(observations, cluster_labels)
    new_cluster_labels = assign_clusters(observations, centroids)

#    print(centroids, cluster_labels, new_cluster_labels)

    # Check if the cluster labels have changed
    if np.array_equal(new_cluster_labels, cluster_labels):
        break
    else:
        cluster_labels = new_cluster_labels


final_clusters = cluster_labels

# Final cluster assignments and centroids
final_clusters, centroids


# In[11]:


### f ###

# Define colors for each cluster
colors = ['orange', 'green']

# Creating the scatter plot
plt.figure(figsize=(8, 6))
for i in range(1, 3):  # Clusters are labeled 1 and 2
    cluster_points = observations[np.array(final_clusters) == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i-1], label=f'Cluster {i}')

plt.title("Scatter Plot of Observations with Cluster Labels")
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
plt.legend()
plt.grid(True)
plt.show()


# In[15]:


'!pip install ISLP'


# In[16]:


############ Problem 8 #########



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ISLP import load_data



USArrests = get_rdataset('USArrests').data

USArrests.head()


# In[17]:


# Sample data to simulate 'USArrests' dataset
np.random.seed(0)
sample_data = np.random.rand(50, 4)  # 50 samples, 4 features

sample_data = USArrests

# Standardize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sample_data)

# Part a: Using the sdev output of the prcomp() function
pca_a = PCA()
pca_a.fit(scaled_data)
pve_a = pca_a.explained_variance_ratio_

pve_a


# In[18]:


# Part b: Applying Equation 12.10 directly
pca_b = PCA()
X_transformed = pca_b.fit_transform(scaled_data)
pve_b = np.sum(X_transformed**2, axis=0) / np.sum(scaled_data**2)

pve_b


# In[44]:


######### Problem 9 ############


import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.datasets import get_rdataset


# a. Hierarchical clustering with complete linkage and Euclidean distance
np.random.seed(42)
hc_uploaded = linkage(USArrests, method='complete')


plt.figure(figsize=(15, 10))
dendrogram(hc_uploaded, labels=sample_data.index, leaf_rotation=90)
plt.title('Dendrogram for USArrests Dataset')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.show()



# In[20]:


# b. Cutting the dendrogram to form three clusters
clusters_uploaded = fcluster(hc_uploaded, 3, criterion='maxclust')
cluster_dict_uploaded = {i: USArrests.index[clusters_uploaded == i].tolist() for i in range(1, 4)}

cluster_dict_uploaded


# In[45]:


# c. Hierarchical clustering after scaling the variables
scaler = StandardScaler()
USArrests_scaled_uploaded = scaler.fit_transform(USArrests)
hc_scaled_uploaded = linkage(USArrests_scaled_uploaded, method='complete')

plt.figure(figsize=(15, 10))
dendrogram(hc_scaled_uploaded, labels=sample_data.index, leaf_rotation=90)
plt.title('Dendrogram for USArrests Dataset (after scaling)')
plt.xlabel('States')
plt.ylabel('Euclidean Distance')
plt.show()


# In[22]:


# d. Cutting the scaled dendrogram to form three clusters
clusters_scaled_uploaded = fcluster(hc_scaled_uploaded, 3, criterion='maxclust')
cluster_scaled_dict_uploaded = {i: USArrests.index[clusters_scaled_uploaded == i].tolist() for i in range(1, 4)}

cluster_scaled_dict_uploaded


# In[23]:


######### Problem 13 ##########


import pandas as pd

# Load the data
file_path = 'Ch12Ex13.csv'
data = pd.read_csv(file_path, header=None)

# Assign column names as specified
data.columns = [f'H{i+1}' if i < 20 else f'D{i-19}' for i in range(0, 40)]

data.head()  # Display the first few rows to verify the loading and naming process


# In[26]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np

# Define a function to perform hierarchical clustering and plot the dendrogram
def plot_hierarchical_clustering(data, method):
    # Calculate the correlation-based distance
    correlation_distance = 1 - np.corrcoef(data.T)

    # Convert full distance matrix to condensed form
    condensed_dist_matrix = squareform(correlation_distance, checks=False)

    # Then use this condensed matrix in the linkage function
    hc = linkage(condensed_dist_matrix, method=method)


    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(hc, labels=data.columns)
    plt.title(f"Hierarchical Clustering Dendrogram ({method} linkage)")
    plt.xlabel("Sample")
    plt.ylabel("Distance")
    plt.show()

# Apply hierarchical clustering using different linkage methods
linkage_methods = ["complete", "average", "single"]
for method in linkage_methods:
    plot_hierarchical_clustering(data, method)


# In[36]:


from scipy.stats import ttest_ind

# Create a label vector for the two groups
labels = ['Healthy']*20 + ['Diseased']*20


# Calculate p-values for each gene
p_values = []
for row in data.iterrows():
    gene_data = row[1]
    t_stat, p_val = ttest_ind(gene_data[:20], gene_data[20:])  # Comparing healthy vs diseased
    # print(gene_data[:20], gene_data[20:])
    p_values.append(p_val)

# Adjust p-values for multiple testing (False Discovery Rate)
adjusted_p_values = pd.Series(p_values).apply(lambda p: min(p * len(p_values), 1.0))

# Find genes with significant difference (threshold: adjusted p-value < 0.05)
significant_genes = adjusted_p_values < 0.05

# Display the indices of the significant genes
significant_gene_indices = significant_genes[significant_genes].index + 1  # Adding 1 to match the gene numbering in the dataset
significant_gene_indices.tolist()


# In[37]:


########## Chapter 13 #############


# In[40]:


########### Problem 4 #############


import numpy as np

# Given p-values
pvals = np.array([0.0011, 0.031, 0.017, 0.32, 0.11, 0.90, 0.07, 0.006, 0.004, 0.0009])
hypotheses = ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10"]

# a. Control the Type I error for each null hypothesis at level α = 0.05
rejected_a = [h for h, p in zip(hypotheses, pvals) if p < 0.05]

# b. Control the FWER at level α = 0.05
alpha_fwer = 0.05 / 10
rejected_b = [h for h, p in zip(hypotheses, pvals) if p < alpha_fwer]

# c. Control the FDR at level q = 0.05
from statsmodels.stats.multitest import multipletests
rejected_c, _, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
rejected_hypotheses_c = [h for h, rej in zip(hypotheses, rejected_c) if rej]

rejected_a, rejected_b, rejected_hypotheses_c



# In[46]:


#################### Problem 8 ####################

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# Set seed for reproducibility
np.random.seed(1)

# Simulate the data
n = 20
m = 100
X = np.random.normal(0, 1, (n, m))

# Perform one-sample t-tests
pvals = np.array([stats.ttest_1samp(X[:, j], 0)[1] for j in range(m)])

# Plot histogram of p-values
plt.hist(pvals, bins=10, edgecolor='black')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.title('Histogram of p-values')
plt.show()

# Type I error control at level α = 0.05
type1_errors = np.sum(pvals < 0.05)

# Control the FWER at level 0.05
fwer_errors = np.sum(pvals < 0.05 / m)

# Control the FDR at level 0.05
fdr_results = multipletests(pvals, alpha=0.05, method='fdr_bh')
fdr_errors = np.sum(fdr_results[0])

# Results for Type I error, FWER, and FDR control
type1_errors, fwer_errors, fdr_errors


# In[47]:


# Cherry-picking the 10 best performing fund managers
# In this case, "best" is interpreted as having the highest total returns over the period
best_performers = np.argsort(np.sum(X, axis=0))[-10:]
pvals_best = pvals[best_performers]

# FWER control for the 10 cherry-picked fund managers (Bonferroni correction)
fwer_best = np.sum(pvals_best < 0.05 / 10)

# FDR control for the 10 cherry-picked fund managers
fdr_best_results = multipletests(pvals_best, alpha=0.05, method='fdr_bh')
fdr_best = np.sum(fdr_best_results[0])

# Results for FWER and FDR control for cherry-picked managers
fwer_best, fdr_best


# In[ ]:




