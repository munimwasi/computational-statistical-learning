# MAS456 — Statistical Computing, Data Analysis, and Machine Learning Portfolio

A comprehensive portfolio of statistical learning, computational data analysis, and predictive modeling projects completed at **KAIST** (Statistical Methods with Computer).

This repository showcases an end-to-end pipeline for processing complex, high-dimensional, and noisy datasets. The projects emphasize both **theoretical statistical rigor** and **hands-on implementation**, focusing on extracting interpretable insights from real-world data to drive structural and systemic analysis.

---

## 📂 Repository Architecture

    Statistical-Methods-with-Computer/
    │
    ├── README.md
    │
    ├── 01_Exploratory_Data_Analysis/
    │   ├── summary_statistics/
    │   ├── correlation_analysis/
    │   └── visualization/
    │
    ├── 02_Statistical_Learning/
    │   ├── linear_regression/
    │   ├── logistic_regression/
    │   └── classification_models/
    │
    ├── 03_Model_Selection_and_Regularization/
    │   ├── ridge_regression/
    │   ├── lasso/
    │   └── cross_validation/
    │
    ├── 04_Resampling_and_Model_Evaluation/
    │   ├── bootstrap/
    │   └── validation_set_approach/
    │
    ├── 05_Tree_Based_and_Ensemble_Methods/
    │   ├── random_forests/
    │   ├── boosting/
    │   └── feature_importance/
    │
    ├── 06_Unsupervised_Learning/
    │   ├── principal_component_analysis/
    │   ├── kmeans_clustering/
    │   └── hierarchical_clustering/
    │
    ├── 07_Survival_Analysis_and_Multiple_Testing/
    │   ├── kaplan_meier/
    │   ├── bonferroni_correction/
    │   └── false_discovery_rate/
    │
    ├── 08_Final_Project_KLIPS_Analysis/
    │   ├── notebooks/
    │   ├── scripts/
    │   └── report/
    │
    └── assets/
        ├── plots/
        ├── heatmaps/
        └── dendrograms/

---

## 🛠️ Technical Skill Matrix

| Domain | Methodologies & Algorithms | Core Applications |
| :--- | :--- | :--- |
| **Statistical Learning** | Linear/Logistic Regression, Bias-Variance Optimization | Parametric modeling, hypothesis testing, system trend tracking |
| **Machine Learning** | KNN, LDA/QDA, Naive Bayes, Decision Trees | Predictive modeling, non-linear classification, boundary isolation |
| **Regularization** | Ridge Regression, LASSO, Cross-Validation | Preventing overfitting, high-dimensional feature selection |
| **Unsupervised Learning** | PCA, K-Means, Hierarchical Clustering | Dimensionality reduction, latent pattern discovery, data segmentation |
| **Advanced Statistics** | Survival Analysis (Kaplan-Meier), Multiple Testing (FDR, Holm) | Analyzing censored event times, maintaining rigorous significance bounds |
| **Computation & Viz** | Correlation Heatmaps, Scatter Matrices, ROC Curves | Exploratory Data Analysis (EDA), scientific data visualization |

---

## 📈 Selected Portfolio Highlights

### 🔹 Regularization & High-Dimensional Predictive Modeling
* **Focus:** Constructing parsimonious predictive models that generalize to unseen, out-of-sample data.
* **Implementation:** Compared Least Squares, Ridge, and LASSO regression using $K$-fold and Validation Set approaches. 
* **Key Insight:** Achieved $\approx 90\%$ variance explanation ($R^2$) on independent test splits for high-dimensional application datasets. Formally demonstrated the mathematical links between $L_1$/$L_2$ penalties and Laplace/Gaussian priors.

### 🔹 Unsupervised Learning & Latent Structure Discovery
* **Focus:** Compressing noisy feature spaces into interpretable low-dimensional representations.
* **Implementation:** Executed Principal Component Analysis (PCA) and Hierarchical Clustering on multi-featured profiles and high-dimensional gene expression data.
* **Key Insight:** Identified that the first 6 Principal Components explained over 70% of total variance in dense datasets. Used hierarchical linkage (Complete, Average, Single) to isolate distinct sub-populations.

### 🔹 Ensemble Methods & Non-Linear Pipelines
* **Focus:** Mitigating the limitations of high-variance estimators through bagging and boosting.
* **Implementation:** Built and tuned Random Forests and Gradient Boosting models, utilizing Gini-impurity for feature importance analysis.
* **Key Insight:** Successfully addressed non-linear interactions in insurance prediction and socioeconomic data, achieving up to a **91.5% AUC** with Random Forest classifiers.

### 🔹 Survival Analysis & Multiple Hypothesis Testing
* **Focus:** Managing temporal data boundaries and maintaining statistical integrity across parallel hypothesis spaces.
* **Implementation:** Designed survival analysis pipelines with Kaplan-Meier estimators and integrated False Discovery Rate (FDR) control.
* **Key Insight:** Applied Holm and Bonferroni procedures to ensure downstream inferences (such as fund manager performance) retained strict statistical significance.

---

## 🏆 Capstone Project: Socioeconomic & Structural Analysis Using KLIPS
### *An End-to-End Pipeline for Pattern Extraction and Under-Reporting Detection*

This project is a data-driven investigation of the **Korean Labor and Income Panel Study (KLIPS)**, focusing on modeling structural inequities and their impact on health outcomes.

* **Algorithmic Pipeline:** Integrated unsupervised segmentation (PCA + K-Means) with penalized models (LASSO, Random Forests).
* **Structural Findings:** Identified age, employment tier, and income quartiles as major predictive signals ($p < 0.05$) for systemic patterns.
* **Reporting Disparities:** The model revealed significant reporting gaps; while only 48.4% of males in the analyzed subset were predicted to have experienced specific structural barriers, the model predicted **90.9% for females**.
* **Systemic Impact:** Statistically confirmed a significant correlation ($p < 0.001$) between these structural experiences and self-rated health status.

---

## 💻 Ecosystem & Tooling

* **Languages:** Python, R, LaTeX *(Technical typesetting)*
* **Scientific Computing:** NumPy, Pandas, SciPy, Statsmodels
* **Machine Learning:** Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

## 🔬 Computational Research Interests & Vision

My work is centered on **scientific data analysis, computational modeling, and machine learning for complex systems**. I am driven by the challenge of translating noisy, multidimensional, real-world data into reproducible and actionable insights.

I possess a deep interest in how large-scale data and rigorous computational modeling can be used to observe and predict complex environmental and societal dynamics. My goal is to apply advanced statistical frameworks—from dimensionality reduction to ensemble modeling—to solve collaborative, high-impact challenges in data-driven research.

---

## 🎓 Academic Profile

* **Author:** Munim Hasan Wasi
* **Affiliation:** SFU (Simon Fraser University)
* **Background:** Alumnus of KAIST (School of Computing)
* **Objective:** This repository demonstrates quantitative research readiness, programming proficiency, and an analytical mindset geared toward collaborative, data-driven exploration.

---
*This repository is intended for educational, research, and portfolio purposes.*
