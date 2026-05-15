# Statistical Computing, Data Analysis, and Machine Learning Portfolio

A comprehensive portfolio of statistical learning, computational data analysis, predictive modeling, and scientific computing projects completed as part of **Statistical Methods with Computer** at **KAIST**.

This repository showcases an end-to-end pipeline of working with complex, high-dimensional, and noisy datasets. The projects contained herein emphasize both **theoretical statistical rigor** and **hands-on computational implementation**, focusing on extracting interpretable insights from real-world data to drive scientific and systemic analysis.

---

## 📂 Repository Architecture

    Statistical-Methods-with-Computer/
    │
    ├── README.md
    │
    ├── 01_Exploratory_Data_Analysis/
    │   ├── summary_statistics/
    │   ├── correlation_analysis/
    │   ├── visualization/
    │   └── feature_relationships/
    │
    ├── 02_Statistical_Learning/
    │   ├── linear_regression/
    │   ├── logistic_regression/
    │   ├── classification_models/
    │   ├── bias_variance_tradeoff/
    │   └── model_interpretation/
    │
    ├── 03_Model_Selection_and_Regularization/
    │   ├── ridge_regression/
    │   ├── lasso/
    │   ├── subset_selection/
    │   ├── cross_validation/
    │   └── bayesian_interpretations/
    │
    ├── 04_Resampling_and_Model_Evaluation/
    │   ├── bootstrap/
    │   ├── validation_set_approach/
    │   ├── k_fold_cross_validation/
    │   └── performance_analysis/
    │
    ├── 05_Tree_Based_and_Ensemble_Methods/
    │   ├── decision_trees/
    │   ├── bagging/
    │   ├── random_forests/
    │   ├── boosting/
    │   └── feature_importance/
    │
    ├── 06_Unsupervised_Learning/
    │   ├── principal_component_analysis/
    │   ├── kmeans_clustering/
    │   ├── hierarchical_clustering/
    │   ├── dimensionality_reduction/
    │   └── high_dimensional_analysis/
    │
    ├── 07_Survival_Analysis_and_Multiple_Testing/
    │   ├── kaplan_meier/
    │   ├── censored_data/
    │   ├── bonferroni_correction/
    │   ├── false_discovery_rate/
    │   └── statistical_significance/
    │
    ├── 08_Final_Project_KLIPS_Analysis/
    │   ├── notebooks/
    │   ├── scripts/
    │   ├── visualizations/
    │   ├── report/
    │   └── presentation/
    │
    └── assets/
        ├── figures/
        ├── plots/
        ├── heatmaps/
        ├── dendrograms/
        └── roc_curves/

---

## 🛠️ Technical Skill Matrix

| Domain | Methodologies & Algorithms | Core Applications |
| :--- | :--- | :--- |
| **Statistical Learning** | Linear/Logistic Regression, Bias-Variance Optimization, Inference | Parametric modeling, hypothesis testing, tracking system trends |
| **Machine Learning** | KNN, LDA/QDA, Naive Bayes, Decision Trees, Ensemble Learning | Predictive modeling, non-linear classification, boundary isolation |
| **Regularization** | Ridge Regression, LASSO, Best Subset Selection, Cross-Validation | Preventing overfitting, high-dimensional feature selection |
| **Unsupervised Learning** | PCA, K-Means, Hierarchical Clustering (Complete/Average/Single) | Dimensionality reduction, latent pattern discovery, data segmentation |
| **Advanced Statistics** | Survival Analysis (Kaplan-Meier), Multiple Testing (FWER, Bonferroni, FDR) | Analyzing censored event times, maintaining rigorous significance bounds |
| **Computation & Viz** | Correlation Heatmaps, Pair Plots, Scatter Matrices, ROC Curves | Exploratory Data Analysis (EDA), scientific data visualization |

---

## 📈 Selected Portfolio Highlights

### 🔹 Exploratory Data Analysis & Feature Engineering
*   **Focus:** Parsing structured multivariate datasets to diagnose system states, isolate trends, and discover hidden correlations.
*   **Implementation:** Developed robust workflows utilizing summary statistics, multi-variable correlation matrices, and pairwise scatterplots.
*   **Key Insight:** Successfully extracted clear signals from complex features (e.g., identifying deterministic vehicle and environmental attributes), mapping the foundational variance distributions prior to modeling.

### 🔹 Regularization & High-Dimensional Predictive Modeling
*   **Focus:** Constructing parsimonious predictive models capable of generalizing effectively to unseen, out-of-sample data.
*   **Implementation:** Implemented and compared Least Squares, Ridge, and LASSO regression models backed by cross-validated training loops ($K$-Fold and Validation Set approaches). 
*   **Key Insight:** Achieved $\approx 90\%$ variance explanation ($R^2$) on independent test splits. Demonstrated the mathematical duality between $L_1$/$L_2$ regularizers and their respective Bayesian priors (Laplace and Gaussian) to justify feature sparsification.

### 🔹 Unsupervised Learning & Latent Structure Discovery
*   **Focus:** Compressing noisy, multi-dimensional feature spaces into interpretable low-dimensional representations without sacrificing key variance metrics.
*   **Implementation:** Executed Principal Component Analysis (PCA) alongside K-Means and Hierarchical Clustering algorithms applied to multi-featured profiles and high-dimensional gene expression data.
*   **Key Insight:** Isolated distinct structural sub-populations using hierarchical linkage metrics (Complete, Average, Single). Used PCA to explain over 70% of total variance across dense features within the first 6 principal components, successfully visualizing latent data patterns.

### 🔹 Ensemble Methods & Non-Linear Pipelines
*   **Focus:** Overcoming the limitations of high-variance estimators when modeling highly non-linear, multi-tiered phenomena.
*   **Implementation:** Built, tuned, and evaluated Decision Trees, Bagged Estimators, Random Forests, and Gradient Boosting pipelines.
*   **Key Insight:** Significantly minimized Out-Of-Bag (OOB) and test Mean Squared Error (MSE) relative to traditional baseline classifiers. Leveraged Gini-impurity and permutation metrics to map structural feature-importance hierarchies.

### 🔹 Survival Analysis & Multiple Hypothesis Testing
*   **Focus:** Handling temporal data boundaries and maintaining statistical integrity across vast, parallel hypothesis spaces.
*   **Implementation:** Designed parametric and non-parametric survival analysis pipelines utilizing Kaplan-Meier estimators to analyze censored data points. Integrated False Discovery Rate (FDR) control and Bonferroni corrections to counteract Family-Wise Error Rate (FWER) inflation.
*   **Key Insight:** Ensured that downstream inferences retained strict statistical significance, providing a robust framework for identifying true physical or structural anomalies while minimizing type-1 errors.

---

## 🏆 Capstone Project: Socioeconomic & Structural Analysis Using KLIPS
### *An End-to-End Pipeline for Pattern Extraction and Under-Reporting Detection*

This repository culminates in a complete, data-driven investigation of the **Korean Labor and Income Panel Study (KLIPS)** dataset, focusing on modeling structural inequities, population demographics, and downstream health-related outcomes.

*   **Algorithmic Pipeline:** Integrated unsupervised segmentation (PCA + K-Means) with robust predictive classification models (Penalized Logistic Regression via LASSO, Random Forests).
*   **Handling Latent Biases:** Addressed inherent data collection bottlenecks (such as the systematic under-reporting of sensitive occurrences) by training ensemble networks to cross-examine reported metrics against demographic indicators.
*   **Performance Metrics:** The final penalized validation models achieved an **AUC of 91.5%**, revealing that structural factors like employment tier, age brackets, and income quartiles served as major predictive signals ($p < 0.05$).
*   **Downstream Deliverables:** The `08_Final_Project_KLIPS_Analysis/` directory contains the complete modular Python code, exploratory notebooks, a comprehensive academic research paper, and a presentation deck detailing the core findings.

---

## 💻 Ecosystem & Tooling

*   **Languages:** Python, R, LaTeX *(for mathematical typesetting and technical reports)*
*   **Scientific Computing Stack:** NumPy, Pandas, SciPy, Statsmodels
*   **Machine Learning Frameworks:** Scikit-Learn
*   **Data Visualization:** Matplotlib, Seaborn
*   **Version Control & Reproducibility:** Git, GitHub, Jupyter Notebooks

---

## 🌊 Computational Research Interests & Vision

My foundational focus centers on **scientific data analysis, computational modeling, and machine learning for complex systems**. I am driven by the process of taking messy, multidimensional, real-world data and translating it into reproducible, interpretable, and actionable scientific insights.

> **Personal Motivation:** As someone originally from Bangladesh — a nation at the front lines of climate change, sea-level rise, erratic flooding, and extreme environmental volatility — I possess a deep-seated interest in how large-scale, automated scientific sensor arrays and rigorous computational modeling can be harnessed to observe, understand, and predict complex environmental dynamics. 
> 
> Whether tracking socioeconomic indicators in dense panel datasets or analyzing multi-parameter spatio-temporal variables from automated physical instruments, my goal is to apply rigorous, cluster-based, and predictive statistical frameworks to solve high-impact, collaborative team-science challenges.

---

## 🎓 Academic Profile

*   **Author:** Munim Hasan Wasi
*   **Affiliation:** SFU (Simon Fraser University)
*   **Objective:** This repository serves as a foundational open-source portfolio demonstrating quantitative research readiness, programming proficiency, and an analytical mindset geared toward collaborative, data-driven scientific exploration.

---
*This repository is fully optimized for educational, scientific portfolio, and open-source verification purposes.*
