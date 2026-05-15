# %%
###### START ######

import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = 'FinalProjectDataset.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset for review
data.head()

# %%
####### QUESTION 1  #######

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# Filter the dataset to include only participants with 'disc_hire' value as 0 or 1
filtered_data = data[data['disc_hire'].isin([0, 1])]

len(filtered_data)

# Define the response variable and explanatory variables
response = filtered_data['disc_hire']
explanatory = filtered_data.drop('disc_hire', axis=1)

# Fit a logistic regression model
logit_model = sm.Logit(response, sm.add_constant(explanatory))
result = logit_model.fit()

# Display the summary of the logistic regression model
result.summary()

# %%
######## QUESTION 2 ########

import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Select the 12 specified variables
selected_columns = ['gender', 'age', 'edu_cat', 'emp_fin', 'income_quartile', 
                    'health', 'disability', 'disc_wage', 'disc_jobedu', 
                    'disc_promotion', 'disc_resign', 'disc_edu']
data_selected = data[selected_columns]
print(data_selected.head())

# %%
# Standardize the data
scaler = StandardScaler()
data_std = scaler.fit_transform(data_selected)

# Apply PCA
pca = PCA()
principalComponents = pca.fit_transform(data_std)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Different Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# %%
# To decide the number of components, look at the plot and identify where the curve starts to plateau
# For instance, if you choose n_components:
n_components = 6 # choose based on the plot

# Re-run PCA with the chosen number of components
pca_final = PCA(n_components=n_components)
principalComponents_final = pca_final.fit_transform(data_std)

# The loadings of the components can be obtained as follows
loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)

# Convert loadings to a DataFrame for easier interpretation
loading_matrix = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)], 
                              index=data_selected.columns)

# Now you can interpret each component based on the loading scores
print(loading_matrix)


# %%
# For each PC, list all the variables that have a loading score of more than 0.4 or less than -0.4

# PC1
def get_important_variables_positive(pc_number):
    pc = loading_matrix[f'PC{pc_number}']
    return pc[pc >= 0.4].index.tolist()

def get_important_variables_negative(pc_number):
    pc = loading_matrix[f'PC{pc_number}']
    return pc[pc <= -0.4].index.tolist()

for i in range(1, n_components + 1):
    print(f'PC{i}:')
    print('Positive:')
    print(get_important_variables_positive(i))
    print('Negative:')
    print(get_important_variables_negative(i))
    print()

# %%
######## QUESTION 3 ########

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use the elbow method to find the optimal number of clusters for the original data
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42,  n_init=10)
    kmeans.fit(data_std)
    wcss.append(kmeans.inertia_)

# Plot the WCSS to visualize the elbow
plt.figure()
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Original Data')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# %%

# Repeat the elbow method for the principal component scores
wcss_pc = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42,  n_init=10)
    kmeans.fit(principalComponents_final)
    wcss_pc.append(kmeans.inertia_)

# Plot the WCSS for the principal components
plt.figure()
plt.plot(range(1, 11), wcss_pc, marker='o', linestyle='--')
plt.title('Elbow Method for Principal Components')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS PC')
plt.show()



# %%
# Calculate the silhouette score for the original data
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42,  n_init=10)
    kmeans.fit(data_std)
    silhouette_scores.append(silhouette_score(data_std, kmeans.labels_))

print(silhouette_scores)

# Plot the silhouette scores
plt.figure()
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Score for Original Data')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# %%
# Calculate the silhouette score for the principal components
silhouette_scores_pc = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42,  n_init=10)
    kmeans.fit(principalComponents_final)
    silhouette_scores_pc.append(silhouette_score(principalComponents_final, kmeans.labels_))

print(silhouette_scores_pc)

# Plot the silhouette scores
plt.figure()
plt.plot(range(2, 11), silhouette_scores_pc, marker='o', linestyle='--')
plt.title('Silhouette Score for Principal Components')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# %%
# Choose the number of clusters (n_clusters) based on the elbow plots
n_clusters = 4# determined from the elbow plot

# Perform K-means clustering with the chosen number of clusters on original data
kmeans_original = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(data_std)

# Perform K-means clustering with the chosen number of clusters on principal components
kmeans_pc = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42,  n_init=10)
clusters_pc = kmeans_pc.fit_predict(principalComponents_final)

# Describe the clusters by examining the centroids
centroids_original = kmeans_original.cluster_centers_
centroids_pc = kmeans_pc.cluster_centers_

# Convert centroids to a DataFrame for easier interpretation
centroids_original_df = pd.DataFrame(scaler.inverse_transform(centroids_original), columns=data_selected.columns)
centroids_pc_df = pd.DataFrame(centroids_pc, columns=[f'PC{i+1}' for i in range(n_components)])

# Now you can describe each cluster by looking at the values of the centroids
print("Centroids for Original Data:")
print(centroids_original_df)
print("\nCentroids for Principal Components:")
print(centroids_pc_df)


# %%
######## QUESTION 4 ########

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve

# Assuming `data` is your full dataset
# Filter out the participants with 'NA' for 'disc_hire' and prepare the data for modeling
data_model = data[data['disc_hire'].notna()]
X = data_model.drop(['disc_hire'], axis=1)
y = data_model['disc_hire'].astype('int')  # Ensure the target is integer

# Splitting data into training and test sets for cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Random Forest Classifier and a grid for cross-validation
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}

# Conduct Grid Search to find the optimal parameters for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='roc_auc')
grid_search_rf.fit(X_train, y_train)
rf_best_params = grid_search_rf.best_params_
rf_best_score = grid_search_rf.best_score_

# Drawing graph for n_estimators and max_depth and see which one produces the best score
n_estimators = [100, 200, 300]
max_depth = [5, 10, 15]
scores = []
for n in n_estimators:
    for d in max_depth:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        scores.append(score)

scores = np.array(scores).reshape(len(n_estimators), len(max_depth))
plt.figure()
plt.imshow(scores, cmap='hot', interpolation='nearest')
plt.xticks(np.arange(len(max_depth)), max_depth)
plt.yticks(np.arange(len(n_estimators)), n_estimators)
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.colorbar()
plt.title('Random Forest Classifier')
plt.show()





# %%

# Define a Penalized Logistic Regression with Lasso penalty
logreg_cv = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', scoring='roc_auc', random_state=42, max_iter=1000)
logreg_cv.fit(X_train, y_train)
logreg_best_score = logreg_cv.scores_[1].mean(axis=0).max()

# Comparing the AUC scores to select the best model
best_model = grid_search_rf.best_estimator_ if rf_best_score > logreg_best_score else logreg_cv

# Print the best model's parameters
print(best_model)

# Print both model's AUC score on the test set
print(f'Random Forest AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])}')
print(f'Logistic Regression AUC: {roc_auc_score(y_test, logreg_cv.predict_proba(X_test)[:, 1])}')

# %%
# Predicting the probabilities on the test set using the best model
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Calculating the ROC curve and determining the optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

optimal_threshold

# %%

# Predicting on the data of the 97 participants who responded 'NA'
NA_data = data[data['disc_hire'].isna()]
NA_pred_prob = best_model.predict_proba(NA_data.drop(['disc_hire'], axis=1))[:, 1]
NA_pred = (NA_pred_prob >= optimal_threshold).astype(int)

# Analyzing the predicted distribution between males and females
NA_data['predicted_disc_hire'] = NA_pred
predicted_distribution = NA_data.groupby('gender')['predicted_disc_hire'].value_counts(normalize=True).unstack()

# This gives you the proportion of predicted 'Yes' responses for hiring discrimination for each gender
print(predicted_distribution)

# %%


# %%
######## QUESTION 5 ########

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

# Assuming `data` contains all participants and `NA_data` contains the 97 participants with 'NA' responses
# Add the predictions to the NA_data
NA_data['health_group'] = 'NA_' + NA_data['predicted_disc_hire'].map({0: 'No', 1: 'Yes'})


# Drop the 'NA' responses from the original data
data_without_na = data[data['disc_hire'].notna()]

# Create the health group for the rest of the data
data_without_na['health_group'] = data_without_na['disc_hire'].map({0: 'No', 1: 'Yes'})

# Combine both datasets
combined_data = pd.concat([data_without_na[['health', 'health_group']], NA_data[['health', 'health_group']]])



# %%
# Conduct an overall ANOVA test
f_value, p_value = stats.f_oneway(combined_data[combined_data['health_group'] == 'No']['health'],
                                   combined_data[combined_data['health_group'] == 'Yes']['health'],
                                   combined_data[combined_data['health_group'] == 'NA_No']['health'],
                                   combined_data[combined_data['health_group'] == 'NA_Yes']['health'])

print(f'ANOVA test results: F-value = {f_value}, p-value = {p_value}')

# If the overall p_value is significant, perform pairwise comparisons
if p_value < 0.05:
    mc = MultiComparison(combined_data['health'], combined_data['health_group'])
    tukey_result = mc.tukeyhsd(alpha=0.05)
    print(tukey_result.summary())
else:
    print("No significant differences were found among the four groups.")


# %%
import matplotlib.pyplot as plt
import numpy as np

# Data for predicted 'disc_hire' responses by gender
gender_categories = ['Male', 'Female']
no_responses = [0.515625, 0.090909]  # Proportion of 'No' responses
yes_responses = [0.484375, 0.909091]  # Proportion of 'Yes' responses

# Number of categories
n_categories = len(gender_categories)

# Creating a bar chart
fig, ax = plt.subplots()

# Setting the positions and width for the bars
bar_width = 0.35
index = np.arange(n_categories)

# Plotting 'No' and 'Yes' responses
bar1 = ax.bar(index, no_responses, bar_width, label='No')
bar2 = ax.bar(index + bar_width, yes_responses, bar_width, label='Yes')

# Adding labels, title and axes ticks
ax.set_xlabel('Gender')
ax.set_ylabel('Proportion')
ax.set_title('Predicted Hiring Discrimination Responses by Gender')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(gender_categories)
ax.legend()

# Displaying the graph
plt.show()


# %%
# Tukey HSD post-hoc test results data
groups = ['NA No vs. NA Yes', 'NA No vs. No', 'NA No vs. Yes', 
          'NA Yes vs. No', 'NA Yes vs. Yes', 'No vs. Yes']
p_values = [0.9526, 0.9744, 0.8013, 0.4964, 0.9874, 0.01]

# Creating a bar chart to visualize the p-values
fig, ax = plt.subplots()

# Bar plot
ax.bar(groups, p_values, color=['blue', 'blue', 'blue', 'blue', 'blue', 'red'])

# Adding labels and title
ax.set_xlabel('Group Comparisons')
ax.set_ylabel('P-value')
ax.set_title('Tukey HSD Post-Hoc Test Results for Self-Rated Health')
# ax.set_yscale('log')  # Log scale due to wide range of p-values
ax.axhline(y=0.05, color='green', linestyle='--')  # Significance level line

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Displaying the graph
plt.show()


# %%
# Data for logistic regression coefficients and p-values
variables = ['gender', 'age', 'edu_cat', 'mariage', 'emp_fin', 'income_quartile',
             'disc_jobedu', 'birth_region', 'health', 'disability', 'residence', 
             'disc_wage', 'disc_promotion', 'disc_resign', 'disc_edu', 
             'disc_home', 'disc_social']
coefficients = [-0.0521, 0.1300, -0.0691, -0.4246, 0.4897, -0.2322, 
                0.1118, -0.1042, 0.1810, 0.2605, 0.0396, 3.2172, 
                0.1990, 0.1593, -0.0154, -0.1022, 1.1763]
p_values = [0.664, 0.037, 0.454, 0.001, 0.000, 0.000, 
            0.325, 0.484, 0.037, 0.429, 0.001, 0.000, 
            0.063, 0.100, 0.914, 0.677, 0.000]




# Creating a bar chart to visualize the p-values
fig, ax = plt.subplots()


# Bar plot

ax.bar(variables, p_values)

# Adding labels and title
ax.set_xlabel('Variables')
ax.set_ylabel('P-value')
ax.set_title('P-values of Logistic Regression Coefficients')
ax.axhline(y=0.05, color='green', linestyle='--')  # Significance level line



# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Displaying the graph
plt.show()


# %%
# list variables which has p-value less than 0.05
significant_variables = [variables[i] for i in range(len(variables)) if p_values[i] < 0.05]
print(significant_variables)


# %%



