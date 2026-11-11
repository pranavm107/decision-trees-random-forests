# Decision Tree & Random Forest Technical Analysis Report

## 1. Dataset & Preprocessing Summary
- **Original Dataset Records**: 1025 rows, 14 columns
- **Identified Exact Duplicates**: 723 rows (70.5%)
- **Clean Deduplicated Cohort**: 302 unique patients
- **Class Distribution (Target)**:
  - Class 0 (No Heart Disease): 138 (45.7%)
  - Class 1 (Heart Disease): 164 (54.3%)
- **Data Leakage Prevention**: Identical duplicate rows were removed prior to train-test splitting so that repeated records do not falsely inflate test metrics.

## 2. Decision Tree Overfitting Analysis
- **Unconstrained Baseline Tree (max_depth=None)**:
  - Training Accuracy: 1.0000 (100% memorization)
  - Testing Accuracy: 0.8033
  - Generalization Gap: 0.1967
- **Optimal Regularized Tree (max_depth=3)**:
  - Constraining tree depth prevents leaf nodes from isolating single training anomalies, maintaining a strong balance between bias and variance.

## 3. Hyperparameter Tuning Results
- **Decision Tree Best Parameters**:
  - Criterion: `gini`
  - Max Depth: `4`
  - Min Samples Split: `10`
  - Min Samples Leaf: `4`
  - Cross-Validation ROC-AUC: `0.8787`

- **Random Forest Best Parameters**:
  - N Estimators: `50`
  - Max Depth: `3`
  - Min Samples Split: `2`
  - Min Samples Leaf: `4`
  - Max Features: `sqrt`
  - Cross-Validation ROC-AUC: `0.9197`

## 4. Top 5 Most Important Clinical Features (Random Forest)
1. **cp** (MDI Importance: 0.2405)
2. **ca** (MDI Importance: 0.1641)
3. **exang** (MDI Importance: 0.1435)
4. **thal** (MDI Importance: 0.1144)
5. **thalach** (MDI Importance: 0.1130)

## 5. 5-Fold Stratified Cross-Validation Summary
- **Tuned Decision Tree**: Mean Accuracy = 0.7714 (+/- 0.0315), Mean ROC-AUC = 0.8515
- **Tuned Random Forest**: Mean Accuracy = 0.8346 (+/- 0.0427), Mean ROC-AUC = 0.9159

## 6. Test Set Performance Summary
| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Decision Tree | 0.8033 | 0.8182 | 0.8182 | 0.8182 | 0.8019 |
| Tuned Decision Tree | 0.7213 | 0.75 | 0.7273 | 0.7385 | 0.8328 |
| Baseline Random Forest | 0.7541 | 0.7647 | 0.7879 | 0.7761 | 0.8615 |
| Tuned Random Forest | 0.7869 | 0.7778 | 0.8485 | 0.8116 | 0.8831 |

## 7. Educational & Clinical Disclaimer
This machine learning project is developed strictly for educational demonstration of tree-based decision models and ensemble techniques. The models have not been clinically validated and must not be used as medical diagnostic tools or substitutes for professional healthcare evaluation.
