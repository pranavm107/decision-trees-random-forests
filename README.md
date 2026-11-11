# Decision Trees and Random Forests: Heart Disease Classification & Generalization Analysis

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

A comprehensive, industry-grade Machine Learning classification project implementing **Decision Trees** and **Random Forests** on the clinical Heart Disease dataset. This repository features systematic investigations into **Gini impurity vs. Entropy (Information Gain)**, **decision tree depth and memorization overfitting**, **hyperparameter optimization with Stratified K-Fold Cross-Validation**, **ensemble variance reduction via bagging and random feature subspaces**, and **impurity vs. permutation feature importance**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objective](#objective)
3. [Dataset Description](#dataset-description)
4. [Target Variable & Class Balance](#target-variable--class-balance)
5. [Technologies Used](#technologies-used)
6. [Project Structure](#project-structure)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Data Preprocessing & Duplicate Investigation](#data-preprocessing--duplicate-investigation)
9. [Decision Tree Implementation](#decision-tree-implementation)
10. [Entropy, Information Gain, and Gini Impurity](#entropy-information-gain-and-gini-impurity)
11. [Decision Tree Visualization](#decision-tree-visualization)
12. [Overfitting & Depth Optimization Analysis](#overfitting--depth-optimization-analysis)
13. [Hyperparameter Tuning](#hyperparameter-tuning)
14. [Random Forest Implementation](#random-forest-implementation)
15. [Feature Importance Analysis](#feature-importance-analysis)
16. [Cross-Validation Stability](#cross-validation-stability)
17. [Model Evaluation & Diagnostic Curves](#model-evaluation--diagnostic-curves)
18. [Model Comparison](#model-comparison)
19. [Installation Instructions](#installation-instructions)
20. [Execution Instructions](#execution-instructions)
21. [Generated Visualizations](#generated-visualizations)
22. [Saved Model Artifacts](#saved-model-artifacts)
23. [Key Findings](#key-findings)
24. [Conclusion & Clinical Disclaimer](#conclusion--clinical-disclaimer)

---

## 1. Project Overview
Tree-based algorithms are fundamental building blocks in applied machine learning. Single decision trees provide high interpretability and intuitive non-linear decision boundaries, but suffer from high variance and a propensity to memorize noise. Random Forests address these limitations by aggregating decorrelated trees through bootstrap sampling and random subspace feature selection.

This project delivers:
- Systematic empirical investigation into decision tree complexity and overfitting.
- Mathematical and practical comparisons between Gini impurity and Shannon Entropy.
- Defense against **data leakage** arising from duplicate records in clinical benchmarks.
- Exhaustive evaluation across Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC.
- Export of production-ready serialized models (`.joblib`) and verification of zero inference drift.

---

## 2. Objective
1. Train a Decision Tree Classifier and visualize its hierarchical split architecture.
2. Analyze the mechanism of overfitting and control tree depth regularizations.
3. Compare Gini Impurity against Information Gain (Entropy).
4. Train a Random Forest ensemble and evaluate variance reduction.
5. Quantify feature contributions using Mean Decrease in Impurity (MDI) and Permutation Importance.
6. Perform 5-fold Stratified Cross-Validation for unbiased stability estimation.
7. Provide production artifacts, reusable scripts, and an interactive Jupyter Notebook.

---

## 3. Dataset Description
The analysis utilizes the clinical Heart Disease dataset (`dataset/heart.csv`), derived from the Cleveland Clinic foundation database:
- **Total Records (Raw)**: 1,025 observations
- **Total Features**: 14 columns (13 clinical predictors + 1 binary target)
- **Missing Values**: 0 null values across all features

### Clinical Predictors:
| Column | Type | Clinical Interpretation | Scale / Values |
|---|---|---|---|
| `age` | Continuous | Patient age in years | 29 to 77 years |
| `sex` | Binary | Biological sex | `1` = Male, `0` = Female |
| `cp` | Nominal | Chest pain type | `0`: Typical Angina, `1`: Atypical Angina, `2`: Non-anginal, `3`: Asymptomatic |
| `trestbps` | Continuous | Resting blood pressure on admission | 94 to 200 mm Hg |
| `chol` | Continuous | Serum cholesterol | 126 to 564 mg/dl |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dl | `1` = True, `0` = False |
| `restecg` | Nominal | Resting electrocardiographic results | `0`: Normal, `1`: ST-T wave abnormality, `2`: Left ventricular hypertrophy |
| `thalach` | Continuous | Maximum heart rate achieved during exercise | 71 to 202 bpm |
| `exang` | Binary | Exercise-induced angina | `1` = Yes, `0` = No |
| `oldpeak` | Continuous | ST depression induced by exercise relative to rest | 0.0 to 6.2 mm |
| `slope` | Ordinal | Peak exercise ST segment slope | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| `ca` | Discrete | Major vessels colored by fluoroscopy | 0 to 4 vessels |
| `thal` | Nominal | Thallium stress scintigraphy | `0`: Null/Normal, `1`: Fixed defect, `2`: Reversible defect, `3`: Normal |

---

## 4. Target Variable & Class Balance
The target variable is `target`:
- `0`: Absence of heart disease (< 50% diameter narrowing in major vessel).
- `1`: Presence of heart disease (> 50% diameter narrowing in major vessel).

In the raw dataset ($N=1,025$):
- **Class 1 (Disease)**: 526 records (51.3%)
- **Class 0 (No Disease)**: 499 records (48.7%)

In the clean deduplicated patient cohort ($N=302$):
- **Class 1 (Disease)**: 164 patients (54.3%)
- **Class 0 (No Disease)**: 138 patients (45.7%)

The dataset exhibits balanced class representation, avoiding severe minority imbalance while requiring stratified splitting.

---

## 5. Technologies Used
- **Language**: Python 3.12
- **Data Manipulation**: Pandas (`2.2.2`), NumPy (`1.26.4`)
- **Machine Learning & Modeling**: Scikit-Learn (`1.4.2`)
- **Model Serialization**: Joblib (`1.4.0`)
- **Data Visualization**: Matplotlib (`3.8.4`), Seaborn (`0.13.2`)
- **Tree Rendering**: Graphviz (`0.21`) & Scikit-learn `plot_tree`
- **Interactive Computing**: Jupyter Lab (`4.2.5`) / Notebook (`7.0.0`)

---

## 6. Project Structure
```
decision-trees-random-forests/
│
├── dataset/
│   └── heart.csv                          # Original untouched clinical dataset
│
├── notebooks/
│   └── decision_trees_random_forests.ipynb# Fully executed end-to-end Jupyter Notebook
│
├── outputs/
│   ├── visualizations/                   # 12 Publication-quality charts & diagrams
│   │   ├── target_distribution.png
│   │   ├── numerical_distributions.png
│   │   ├── categorical_distributions.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_target_relationships.png
│   │   ├── decision_tree.png
│   │   ├── decision_tree.dot
│   │   ├── decision_tree_depth_analysis.png
│   │   ├── random_forest_feature_importance.png
│   │   ├── cross_validation_comparison.png
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── random_forest_confusion_matrix.png
│   │   └── model_roc_comparison.png
│   │
│   ├── models/                           # Serialized production models
│   │   ├── decision_tree_classifier.joblib
│   │   └── random_forest_classifier.joblib
│   │
│   └── reports/                          # Metric summaries and CSV tabular reports
│       ├── model_comparison.csv
│       ├── cross_validation_results.csv
│       ├── feature_importance.csv
│       ├── classification_metrics.csv
│       └── decision_tree_analysis.md
│
├── src/
│   ├── __init__.py
│   ├── train_and_evaluate.py             # Reusable end-to-end Python pipeline
│   └── build_and_run_notebook.py         # Automated notebook compilation script
│
├── README.md                             # Comprehensive project documentation
├── requirements.txt                      # Reproducible pinned dependencies
├── .gitignore                            # Git exclusion rules
└── LICENSE                               # MIT License
```

---

## 7. Exploratory Data Analysis (EDA)

### Continuous Feature Distributions
- **`age`**: Centered around 54 years with near-Gaussian distribution (range: 29-77).
- **`trestbps`**: Moderate right skewness, peak around 120-130 mm Hg.
- **`chol`**: Pronounced right tail with several extreme values (> 350 mg/dl).
- **`thalach`**: Negative skewness, with disease-positive patients demonstrating higher peak exercise heart rates.
- **`oldpeak`**: Marked concentration at 0.0 (no depression) with right skew.

### Categorical Feature Stratification
- Patients with `cp` $\in \{1, 2, 3\}$ (atypical, non-anginal, or asymptomatic chest pain) exhibit significantly higher incidence of heart disease compared to typical angina (`cp = 0`).
- Exercise-induced angina (`exang = 1`) strongly correlates with absence of disease in this cohort (`target = 0`), while `ca = 0` (zero colored vessels) correlates heavily with `target = 1`.

---

## 8. Data Preprocessing & Duplicate Investigation

### The Duplicate Record Problem & Data Leakage
The raw dataset contains **723 duplicate rows** out of 1,025 (70.54%). A deep inspection revealed:
1. **Identical Observations**: Duplicates are identical across all 14 columns.
2. **Zero Conflicting Labels**: No contradictory labels exist for identical feature sets.
3. **Cleveland Origin**: The Cleveland benchmark contains 303 patient records. The 1,025-record file is an oversampled artifact.
4. **Data Leakage Hazard**: If duplicate records span both training and test sets, models memorize duplicate patient instances, producing deceptive test accuracies exceeding 98%.

### Leakage-Safe Strategy
We preserve `dataset/heart.csv` intact, remove duplicate records to isolate the genuine cohort of **302 unique patients**, and conduct all subsequent modeling on this leakage-free dataset.

### Cohort Audit Summary:
- Original Records: `1,025`
- Duplicate Records Removed: `723`
- Remaining Unique Patients: `302`
- Target Distribution Before: Class 0 = 48.7%, Class 1 = 51.3%
- Target Distribution After: Class 0 = 45.7%, Class 1 = 54.3%

---

## 9. Decision Tree Implementation
We train an initial unconstrained Decision Tree on 80% stratified training data ($N=241$):
```python
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)
```
- **Depth Reached**: 9 levels
- **Leaves Formed**: 35 terminal nodes
- **Training Accuracy**: **1.0000 (100% memorization)**
- **Testing Accuracy**: **0.8033 (80.33%)**
- **Generalization Gap**: **0.1967 (19.67% error drop)**

The unconstrained tree partitions feature space until every leaf is pure ($I_G = 0$), memorizing training-specific noise.

---

## 10. Entropy, Information Gain, and Gini Impurity

### Mathematical Comparison
For binary classification at node $m$ with class proportions $p_0, p_1$:

$$\text{Gini Impurity: } I_G(m) = 1 - (p_0^2 + p_1^2)$$

$$\text{Shannon Entropy: } H(m) = - (p_0 \log_2 p_0 + p_1 \log_2 p_1)$$

$$\text{Information Gain: } IG(T, a) = H(T) - \sum_{v} \frac{|T_v|}{|T|} H(T_v)$$

### Empirical Results:
| Criterion | Test Accuracy | Test ROC-AUC | Max Depth | Leaves |
|---|---|---|---|---|
| **Gini Impurity** | **0.8033** | **0.8019** | 9 | 35 |
| **Entropy (Info Gain)** | **0.7213** | **0.7208** | 9 | 34 |

Gini impurity is computationally faster (avoiding logarithmic calls) and achieved superior test accuracy and split stability on this clinical dataset.

---

## 11. Decision Tree Visualization
The learned tree is visualized using `sklearn.tree.plot_tree` (pruned to depth 3 for clinical legibility) and exported as `decision_tree.dot`:
- **Root Node Split**: `cp <= 0.5` (separating typical angina from other chest pain variants).
- **Subsequent Splitting Features**: `ca` (colored vessels), `thalach` (max heart rate), and `oldpeak` (ST depression).
- Each node visualizes: split condition, impurity metric, sample count, class distribution, and predicted majority label.

---

## 12. Overfitting & Depth Optimization Analysis
To evaluate model capacity against generalization error, we evaluated trees constrained across depths 1 through 10 and unconstrained (`None`):

| `max_depth` | Training Accuracy | Testing Accuracy | Generalization Gap | Status |
|:---:|:---:|:---:|:---:|---|
| **1** | 0.7842 | 0.6557 | 0.1285 | Underfitting (High Bias) |
| **2** | 0.7842 | 0.6557 | 0.1285 | Underfitting (High Bias) |
| **3** | **0.8589** | **0.8033** | **0.0556** | **Optimal Trade-Off Zone** |
| **4** | 0.8714 | 0.7213 | 0.1501 | Moderate Overfitting |
| **5** | 0.9419 | 0.7705 | 0.1714 | Severe Overfitting |
| **6** | 0.9627 | 0.7705 | 0.1922 | Severe Overfitting |
| **7** | 0.9793 | 0.7705 | 0.2088 | Severe Overfitting |
| **8** | 0.9959 | 0.7869 | 0.2090 | Memorization |
| **9** | 1.0000 | 0.8033 | 0.1967 | Complete Memorization |
| **10** | 1.0000 | 0.8033 | 0.1967 | Complete Memorization |
| **None** | 1.0000 | 0.8033 | 0.1967 | Complete Memorization |

**Key Finding**: Depth 3 produces the smallest generalization gap (0.0556) while capturing primary clinical interactions without memorizing sample idiosyncrasies.

---

## 13. Hyperparameter Tuning

### Decision Tree Optimization (`GridSearchCV`):
- **Search Space**: `criterion` (`gini`, `entropy`), `max_depth` (`[2..8, None]`), `min_samples_split` (`[2, 5, 10, 15]`), `min_samples_leaf` (`[1, 2, 4, 6]`).
- **Optimal Hyperparameters**:
  - `criterion`: `'gini'`
  - `max_depth`: `4`
  - `min_samples_leaf`: `4`
  - `min_samples_split`: `10`
- **Best Training CV ROC-AUC**: `0.8787`

### Random Forest Optimization (`GridSearchCV`):
- **Search Space**: `n_estimators` (`[50, 100, 150]`), `max_depth` (`[3, 5, 8, None]`), `min_samples_split` (`[2, 5, 10]`), `min_samples_leaf` (`[1, 2, 4]`), `max_features` (`['sqrt', 'log2']`).
- **Optimal Hyperparameters**:
  - `n_estimators`: `50`
  - `max_depth`: `3`
  - `min_samples_leaf`: `4`
  - `min_samples_split`: `2`
  - `max_features`: `'sqrt'`
- **Best Training CV ROC-AUC**: `0.9197`

---

## 14. Random Forest Implementation
Random Forests aggregate predictions from $B$ decorrelated decision trees $\{T_b\}_{b=1}^B$:
1. **Bagging**: Each tree samples $N$ instances with replacement ($\approx 63.2\%$ unique samples per tree).
2. **Random Subspace**: At each split, only $m = \sqrt{p} = \sqrt{13} \approx 3$ features are considered.
3. **Voting**: Aggregation dampens individual tree variance:
   $$\hat{p}(y=1|x) = \frac{1}{B} \sum_{b=1}^B \hat{p}_b(y=1|x)$$

---

## 15. Feature Importance Analysis
We quantify clinical contribution via **Mean Decrease in Impurity (MDI)** and validate on the test set via **Permutation Importance**:

| Rank | Clinical Feature | MDI Importance | Permutation Mean | Permutation Std | Description |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | `cp` | **0.2405** | -0.0033 | 0.0191 | Chest Pain Type |
| **2** | `ca` | **0.1641** | +0.0475 | 0.0200 | Major Fluoroscopy Vessels |
| **3** | `exang` | **0.1435** | +0.0197 | 0.0230 | Exercise Induced Angina |
| **4** | `thal` | **0.1144** | +0.0361 | 0.0098 | Thalassemia Defect |
| **5** | `thalach` | **0.1130** | +0.0082 | 0.0361 | Max Heart Rate Achieved |
| **6** | `oldpeak` | **0.0780** | +0.0344 | 0.0279 | ST Depression |
| **7** | `age` | **0.0398** | -0.0016 | 0.0115 | Patient Age |
| **8** | `sex` | **0.0324** | +0.0246 | 0.0223 | Biological Sex |
| **9** | `chol` | **0.0277** | -0.0082 | 0.0110 | Serum Cholesterol |
| **10** | `slope` | **0.0260** | +0.0344 | 0.0115 | Peak Exercise ST Slope |
| **11** | `restecg` | **0.0088** | -0.0066 | 0.0080 | Resting ECG Abnormality |
| **12** | `trestbps` | **0.0087** | -0.0148 | 0.0049 | Resting Blood Pressure |
| **13** | `fbs` | **0.0031** | +0.0000 | 0.0073 | Fasting Blood Sugar |

> **Interpretability Note**: High impurity reduction indicates strong predictive utility in this dataset, not causal clinical pathology.

---

## 16. Cross-Validation Stability
We evaluated 5-Fold Stratified Cross-Validation on the full deduplicated patient cohort:

| Model | Mean CV Accuracy | CV Accuracy Std | Mean CV ROC-AUC | CV ROC-AUC Std |
|---|:---:|:---:|:---:|:---:|
| **Tuned Decision Tree** | 0.7714 | 0.0315 | 0.8515 | 0.0092 |
| **Tuned Random Forest** | **0.8346** | 0.0427 | **0.9159** | 0.0116 |

Random Forest demonstrates a **+6.32% improvement in Mean CV Accuracy** and a **+0.0644 increase in Mean CV ROC-AUC** over the single Decision Tree.

---

## 17. Model Evaluation & Diagnostic Curves

### Test Set Metrics ($N=61$ Patients: 28 No Disease, 33 Disease)
- **Tuned Decision Tree**:
  - Test Accuracy: `0.7213`
  - Precision: `0.7500`
  - Recall: `0.7273` (24 TP / 33 True Positives)
  - F1-Score: `0.7385`
  - ROC-AUC: `0.8328`
  - False Negatives: **9 missed patients**

- **Tuned Random Forest**:
  - Test Accuracy: `0.7869`
  - Precision: `0.7778`
  - Recall: **`0.8485`** (28 TP / 33 True Positives)
  - F1-Score: `0.8116`
  - ROC-AUC: **`0.8831`**
  - False Negatives: **5 missed patients (44.4% reduction in critical false negatives)**

---

## 18. Model Comparison
Summary of all models evaluated on identical test partitions:

| Model | Train Acc | Test Acc | Precision | Recall | F1-Score | ROC-AUC | CV Mean Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Decision Tree** | 1.0000 | 0.8033 | 0.8182 | 0.8182 | 0.8182 | 0.8019 | 0.7748 |
| **Tuned Decision Tree** | 0.8714 | 0.7213 | 0.7500 | 0.7273 | 0.7385 | 0.8328 | 0.7714 |
| **Baseline Random Forest** | 1.0000 | 0.7541 | 0.7647 | 0.7879 | 0.7761 | 0.8615 | 0.8247 |
| **Tuned Random Forest** | 0.8921 | **0.7869** | **0.7778** | **0.8485** | **0.8116** | **0.8831** | **0.8346** |

---

## 19. Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment tool (`venv` or `conda`)

```bash
# Clone the repository
git clone https://github.com/pranavm107/decision-trees-random-forests.git
cd decision-trees-random-forests

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 20. Execution Instructions

### Execute Full Pipeline Script:
```bash
python3 src/train_and_evaluate.py
```

### Launch Interactive Jupyter Notebook:
```bash
jupyter notebook notebooks/decision_trees_random_forests.ipynb
```

---

## 21. Generated Visualizations
All charts are exported to `outputs/visualizations/`:
1. `target_distribution.png`: Raw vs. clean class balance.
2. `numerical_distributions.png`: Continuous physiological variables.
3. `categorical_distributions.png`: Stratified risk bar plots.
4. `correlation_heatmap.png`: Correlation matrix.
5. `feature_target_relationships.png`: Key diagnostic boxplots.
6. `decision_tree.png`: Hierarchical tree topology.
7. `decision_tree_depth_analysis.png`: Overfitting depth curve.
8. `random_forest_feature_importance.png`: MDI ranking chart.
9. `cross_validation_comparison.png`: Fold-by-fold accuracy bar chart.
10. `decision_tree_confusion_matrix.png`: Diagnostic confusion matrix.
11. `random_forest_confusion_matrix.png`: Ensemble confusion matrix.
12. `model_roc_comparison.png`: Multi-model ROC curves.

---

## 22. Saved Model Artifacts
Trained models are saved in `outputs/models/`:
- `decision_tree_classifier.joblib` (3.9 KB)
- `random_forest_classifier.joblib` (79.8 KB)

### Verification:
Reloaded models were tested against out-of-sample data, confirming 100% identical inference.

---

## 23. Key Findings
1. **Severe Data Leakage Risk**: 70.5% duplicate rows in the raw dataset falsely inflated accuracy when not deduplicated. Deduplication is vital for valid validation.
2. **Decision Tree Memorization**: Unconstrained trees easily achieve 100% training accuracy through node memorization. Restricting depth to 3-4 eliminates this gap.
3. **Random Forest Sensitivity**: Tuned Random Forest reduced false negatives from 9 to 5, boosting sensitivity to 84.85% and ROC-AUC to 0.8831.
4. **Discriminative Biomarkers**: Chest pain type (`cp`), fluoroscopy vessels (`ca`), exercise angina (`exang`), and thallium defects (`thal`) dominated predictive importance.

---

## 24. Conclusion & Clinical Disclaimer

### Conclusion
This project successfully implemented, visualized, and compared Decision Trees and Random Forests on the Heart Disease dataset. By removing duplicate data leakage, investigating depth-based overfitting, comparing Gini and Entropy, and tuning ensemble hyperparameters with 5-fold cross-validation, the project establishes a robust, reproducible machine learning benchmark.

### Clinical Disclaimer
> **IMPORTANT NOTICE**: This software and documentation are created solely for academic and educational demonstrations of supervised machine learning techniques. The models have **not** been clinically validated and are **not** certified diagnostic tools. They must **never** be used for medical decisions, diagnosis, or treatment. Consult a licensed medical practitioner for all clinical decisions.

---

## License
This project is licensed under the [MIT License](LICENSE).

