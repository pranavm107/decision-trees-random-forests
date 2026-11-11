"""
End-to-end Machine Learning Pipeline for Decision Trees and Random Forests
Heart Disease Classification
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Configure styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'heart.csv')
VIS_DIR = os.path.join(BASE_DIR, 'outputs', 'visualizations')
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STARTING DECISION TREE & RANDOM FOREST PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Load Data
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Duplicate analysis
    n_raw = len(df_raw)
    n_dups = df_raw.duplicated().sum()
    print(f"Original row count: {n_raw}")
    print(f"Duplicate rows: {n_dups} ({n_dups/n_raw*100:.2f}%)")
    print(f"Raw target distribution:\n{df_raw['target'].value_counts(normalize=True)}")

    # Deduplicate for clean leakage-safe modeling
    df = df_raw.drop_duplicates().reset_index(drop=True)
    n_clean = len(df)
    print(f"Clean unique patient records: {n_clean}")
    print(f"Clean target distribution:\n{df['target'].value_counts(normalize=True)}")

    # 2. Exploratory Data Analysis & Visualizations
    # A. Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['target'].value_counts()
    percentages = df['target'].value_counts(normalize=True) * 100
    bars = ax.bar(['No Disease (0)', 'Disease (1)'], counts, color=['#3498db', '#e74c3c'], edgecolor='black', width=0.55)
    for bar, pct, cnt in zip(bars, percentages, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f"{cnt} ({pct:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Target Class Distribution (Unique Patients, N=302)', fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Clinical Diagnosis (Target)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_ylim(0, max(counts) + 25)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'target_distribution.png'))
    plt.close()

    # B. Numerical Distributions
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color='#2980b9', edgecolor='black', bins=20)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[5].axis('off')  # hide unused subplot
    plt.suptitle('Distributions of Continuous Clinical Measurements', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'numerical_distributions.png'))
    plt.close()

    # C. Categorical Distributions
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for idx, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, hue='target', ax=axes[idx], palette=['#3498db', '#e74c3c'])
        axes[idx].set_title(f'{col} by Target', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        if idx == 0:
            axes[idx].legend(['No Disease', 'Disease'], title='Target', loc='upper right')
        else:
            axes[idx].legend([], frameon=False)
    plt.suptitle('Categorical Clinical Features Stratified by Target', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'categorical_distributions.png'))
    plt.close()

    # D. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask, linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Correlation Matrix of Clinical Features and Target', fontsize=15, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'correlation_heatmap.png'))
    plt.close()

    # E. Feature-Target Relationships
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(x='target', y='thalach', data=df, ax=axes[0], palette=['#3498db', '#e74c3c'])
    axes[0].set_title('Max Heart Rate (thalach) vs Target', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(['No Disease', 'Disease'])
    
    sns.boxplot(x='target', y='oldpeak', data=df, ax=axes[1], palette=['#3498db', '#e74c3c'])
    axes[1].set_title('ST Depression (oldpeak) vs Target', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(['No Disease', 'Disease'])

    sns.countplot(x='cp', hue='target', data=df, ax=axes[2], palette=['#3498db', '#e74c3c'])
    axes[2].set_title('Chest Pain Type (cp) vs Target', fontsize=12, fontweight='bold')
    axes[2].legend(['No Disease', 'Disease'], title='Target')
    axes[2].set_xlabel('Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)')

    plt.suptitle('Key Discriminative Clinical Indicators', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'feature_target_relationships.png'))
    plt.close()

    # 3. Train-Test Split
    X = df.drop(columns=['target'])
    y = df['target']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Train class balance: {np.bincount(y_train)}, Test class balance: {np.bincount(y_test)}")

    # 4. Baseline Decision Tree
    dt_baseline = DecisionTreeClassifier(random_state=42)
    dt_baseline.fit(X_train, y_train)
    dt_base_train_acc = accuracy_score(y_train, dt_baseline.predict(X_train))
    dt_base_test_acc = accuracy_score(y_test, dt_baseline.predict(X_test))
    print(f"Baseline Decision Tree -> Train Acc: {dt_base_train_acc:.4f}, Test Acc: {dt_base_test_acc:.4f}")

    # 5. Tree Visualization
    # Visualization of pruned/readable depth tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_baseline,
        max_depth=3,
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=True
    )
    plt.title('Decision Tree Classifier Visualization (Pruned to Depth 3 for Interpretability)', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'decision_tree.png'))
    plt.close()

    # Also export DOT if graphviz is available
    try:
        from sklearn.tree import export_graphviz
        dot_path = os.path.join(VIS_DIR, 'decision_tree.dot')
        export_graphviz(
            dt_baseline,
            out_file=dot_path,
            max_depth=3,
            feature_names=feature_names,
            class_names=['No Disease', 'Disease'],
            filled=True,
            rounded=True,
            special_characters=True
        )
        print("Exported DOT representation to outputs/visualizations/decision_tree.dot")
    except Exception as e:
        print(f"DOT export skipped: {e}")

    # 6. Gini vs Entropy Analysis
    dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42).fit(X_train, y_train)
    dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42).fit(X_train, y_train)
    print(f"Gini Criterion -> Test Acc: {accuracy_score(y_test, dt_gini.predict(X_test)):.4f}, Test ROC-AUC: {roc_auc_score(y_test, dt_gini.predict_proba(X_test)[:, 1]):.4f}")
    print(f"Entropy Criterion -> Test Acc: {accuracy_score(y_test, dt_entropy.predict(X_test)):.4f}, Test ROC-AUC: {roc_auc_score(y_test, dt_entropy.predict_proba(X_test)[:, 1]):.4f}")

    # 7. Overfitting and Tree Depth Optimization Analysis
    depth_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]
    depth_records = []
    for d in depth_candidates:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, model.predict(X_train))
        te_acc = accuracy_score(y_test, model.predict(X_test))
        depth_name = str(d) if d is not None else 'None (Full)'
        depth_records.append({
            'max_depth': depth_name,
            'train_accuracy': tr_acc,
            'test_accuracy': te_acc,
            'gap': tr_acc - te_acc
        })
    df_depth = pd.DataFrame(depth_records)
    print("\nDecision Tree Depth Analysis:")
    print(df_depth.to_string(index=False))

    # Plot Overfitting Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    x_ticks = [str(r['max_depth']) for r in depth_records]
    train_scores = [r['train_accuracy'] for r in depth_records]
    test_scores = [r['test_accuracy'] for r in depth_records]
    ax.plot(x_ticks, train_scores, marker='o', linewidth=2.5, color='#e74c3c', label='Training Accuracy')
    ax.plot(x_ticks, test_scores, marker='s', linewidth=2.5, color='#2980b9', label='Testing Accuracy')
    ax.axvline(x='3', color='green', linestyle='--', linewidth=1.5, label='Optimal Generalization Zone (Depth 3)')
    ax.set_title('Decision Tree Overfitting Analysis: Accuracy vs. Maximum Tree Depth', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Maximum Tree Depth (max_depth)', fontsize=12)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_ylim(0.55, 1.05)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'decision_tree_depth_analysis.png'))
    plt.close()

    # 8. Decision Tree Hyperparameter Tuning with GridSearchCV
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [2, 3, 4, 5, 6, 8, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=dt_param_grid,
        cv=cv_kfold,
        scoring='roc_auc',
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    dt_best = dt_grid.best_estimator_
    print(f"\nBest Decision Tree Parameters: {dt_grid.best_params_}")
    print(f"Best CV ROC-AUC: {dt_grid.best_score_:.4f}")

    # 9. Baseline Random Forest Classifier
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_base_train_acc = accuracy_score(y_train, rf_baseline.predict(X_train))
    rf_base_test_acc = accuracy_score(y_test, rf_baseline.predict(X_test))
    print(f"Baseline Random Forest -> Train Acc: {rf_base_train_acc:.4f}, Test Acc: {rf_base_test_acc:.4f}")

    # 10. Random Forest Hyperparameter Tuning
    rf_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=cv_kfold,
        scoring='roc_auc',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    print(f"\nBest Random Forest Parameters: {rf_grid.best_params_}")
    print(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

    # 11. Feature Importance Analysis
    importances = rf_best.feature_importances_
    perm_importance = permutation_importance(rf_best, X_test, y_test, n_repeats=10, random_state=42)
    
    df_fi = pd.DataFrame({
        'Feature': feature_names,
        'MDI_Importance': importances,
        'Permutation_Mean': perm_importance.importances_mean,
        'Permutation_Std': perm_importance.importances_std
    }).sort_values('MDI_Importance', ascending=False).reset_index(drop=True)
    df_fi['Rank'] = df_fi.index + 1
    df_fi.to_csv(os.path.join(REPORTS_DIR, 'feature_importance.csv'), index=False)
    print("\nFeature Importance Table:")
    print(df_fi.to_string(index=False))

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_fi, x='MDI_Importance', y='Feature', palette='viridis', edgecolor='black')
    plt.title('Random Forest Feature Importance (Mean Decrease in Impurity)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score (Normalized Impurity Reduction)', fontsize=12)
    plt.ylabel('Clinical Feature', fontsize=12)
    for i, v in enumerate(df_fi['MDI_Importance']):
        plt.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=10, fontweight='bold')
    plt.xlim(0, max(df_fi['MDI_Importance']) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'random_forest_feature_importance.png'))
    plt.close()

    # 12. Cross-Validation Comparison
    # Evaluate 5-fold CV for Decision Tree and Random Forest on the clean training set (or entire clean dataset)
    # Using the standard 5-fold CV on clean dataset for overall unbiased stability check
    dt_cv_acc = cross_val_score(dt_best, X, y, cv=cv_kfold, scoring='accuracy')
    dt_cv_roc = cross_val_score(dt_best, X, y, cv=cv_kfold, scoring='roc_auc')

    rf_cv_acc = cross_val_score(rf_best, X, y, cv=cv_kfold, scoring='accuracy')
    rf_cv_roc = cross_val_score(rf_best, X, y, cv=cv_kfold, scoring='roc_auc')

    cv_df = pd.DataFrame([
        {
            'Model': 'Tuned Decision Tree',
            'Mean CV Accuracy': dt_cv_acc.mean(),
            'CV Accuracy Std': dt_cv_acc.std(),
            'Mean CV ROC-AUC': dt_cv_roc.mean(),
            'CV ROC-AUC Std': dt_cv_roc.std(),
            'Fold 1 Acc': dt_cv_acc[0],
            'Fold 2 Acc': dt_cv_acc[1],
            'Fold 3 Acc': dt_cv_acc[2],
            'Fold 4 Acc': dt_cv_acc[3],
            'Fold 5 Acc': dt_cv_acc[4]
        },
        {
            'Model': 'Tuned Random Forest',
            'Mean CV Accuracy': rf_cv_acc.mean(),
            'CV Accuracy Std': rf_cv_acc.std(),
            'Mean CV ROC-AUC': rf_cv_roc.mean(),
            'CV ROC-AUC Std': rf_cv_roc.std(),
            'Fold 1 Acc': rf_cv_acc[0],
            'Fold 2 Acc': rf_cv_acc[1],
            'Fold 3 Acc': rf_cv_acc[2],
            'Fold 4 Acc': rf_cv_acc[3],
            'Fold 5 Acc': rf_cv_acc[4]
        }
    ])
    cv_df.to_csv(os.path.join(REPORTS_DIR, 'cross_validation_results.csv'), index=False)
    print("\nCross-Validation Results:")
    print(cv_df[['Model', 'Mean CV Accuracy', 'CV Accuracy Std', 'Mean CV ROC-AUC', 'CV ROC-AUC Std']].to_string(index=False))

    # Cross-Validation Visual Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    cv_plot_data = pd.DataFrame({
        'Fold': list(range(1, 6)) * 2,
        'Accuracy': list(dt_cv_acc) + list(rf_cv_acc),
        'Model': ['Decision Tree'] * 5 + ['Random Forest'] * 5
    })
    sns.barplot(x='Fold', y='Accuracy', hue='Model', data=cv_plot_data, palette=['#e67e22', '#27ae60'], ax=ax, edgecolor='black')
    ax.axhline(dt_cv_acc.mean(), color='#e67e22', linestyle='--', label=f'DT Mean: {dt_cv_acc.mean():.3f}')
    ax.axhline(rf_cv_acc.mean(), color='#27ae60', linestyle='--', label=f'RF Mean: {rf_cv_acc.mean():.3f}')
    ax.set_title('5-Fold Stratified Cross-Validation Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Accuracy Score', fontsize=12)
    ax.set_xlabel('Cross-Validation Fold', fontsize=12)
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'cross_validation_comparison.png'))
    plt.close()

    # 13. Test Set Evaluation & Metrics
    models = {
        'Baseline Decision Tree': dt_baseline,
        'Tuned Decision Tree': dt_best,
        'Baseline Random Forest': rf_baseline,
        'Tuned Random Forest': rf_best
    }

    comparison_records = []
    classification_metrics_records = []

    for name, m in models.items():
        tr_pred = m.predict(X_train)
        te_pred = m.predict(X_test)
        te_prob = m.predict_proba(X_test)[:, 1]

        tr_acc = accuracy_score(y_train, tr_pred)
        te_acc = accuracy_score(y_test, te_pred)
        prec = precision_score(y_test, te_pred)
        rec = recall_score(y_test, te_pred)
        f1 = f1_score(y_test, te_pred)
        roc_auc = roc_auc_score(y_test, te_prob)
        cv_mean = cross_val_score(m, X, y, cv=cv_kfold, scoring='accuracy').mean()

        comparison_records.append({
            'Model': name,
            'Train Accuracy': round(tr_acc, 4),
            'Test Accuracy': round(te_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
            'CV Mean Accuracy': round(cv_mean, 4)
        })

        classification_metrics_records.append({
            'Model': name,
            'Accuracy': round(te_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })

    df_comp = pd.DataFrame(comparison_records)
    df_metrics = pd.DataFrame(classification_metrics_records)
    df_comp.to_csv(os.path.join(REPORTS_DIR, 'model_comparison.csv'), index=False)
    df_metrics.to_csv(os.path.join(REPORTS_DIR, 'classification_metrics.csv'), index=False)
    print("\nModel Performance Comparison:")
    print(df_comp.to_string(index=False))

    # Confusion Matrix Visualizations
    cm_dt = confusion_matrix(y_test, dt_best.predict(X_test))
    cm_rf = confusion_matrix(y_test, rf_best.predict(X_test))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['No Disease (0)', 'Disease (1)'],
                yticklabels=['No Disease (0)', 'Disease (1)'], annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Tuned Decision Tree: Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Clinical Label', fontsize=11)
    ax.set_ylabel('True Clinical Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'decision_tree_confusion_matrix.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax,
                xticklabels=['No Disease (0)', 'Disease (1)'],
                yticklabels=['No Disease (0)', 'Disease (1)'], annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Tuned Random Forest: Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Clinical Label', fontsize=11)
    ax.set_ylabel('True Clinical Label', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'random_forest_confusion_matrix.png'))
    plt.close()

    # ROC Curve Comparison
    fpr_dt_base, tpr_dt_base, _ = roc_curve(y_test, dt_baseline.predict_proba(X_test)[:, 1])
    fpr_dt_tune, tpr_dt_tune, _ = roc_curve(y_test, dt_best.predict_proba(X_test)[:, 1])
    fpr_rf_base, tpr_rf_base, _ = roc_curve(y_test, rf_baseline.predict_proba(X_test)[:, 1])
    fpr_rf_tune, tpr_rf_tune, _ = roc_curve(y_test, rf_best.predict_proba(X_test)[:, 1])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr_dt_base, tpr_dt_base, color='#e67e22', linestyle=':', linewidth=2, label=f"Baseline DT (AUC = {roc_auc_score(y_test, dt_baseline.predict_proba(X_test)[:, 1]):.3f})")
    ax.plot(fpr_dt_tune, tpr_dt_tune, color='#d35400', linewidth=2.5, label=f"Tuned DT (AUC = {roc_auc_score(y_test, dt_best.predict_proba(X_test)[:, 1]):.3f})")
    ax.plot(fpr_rf_base, tpr_rf_base, color='#3498db', linestyle=':', linewidth=2, label=f"Baseline RF (AUC = {roc_auc_score(y_test, rf_baseline.predict_proba(X_test)[:, 1]):.3f})")
    ax.plot(fpr_rf_tune, tpr_rf_tune, color='#27ae60', linewidth=2.5, label=f"Tuned RF (AUC = {roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1]):.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance (AUC = 0.500)')
    ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'model_roc_comparison.png'))
    plt.close()

    # 14. Save Models with Joblib
    dt_model_path = os.path.join(MODELS_DIR, 'decision_tree_classifier.joblib')
    rf_model_path = os.path.join(MODELS_DIR, 'random_forest_classifier.joblib')
    joblib.dump(dt_best, dt_model_path)
    joblib.dump(rf_best, rf_model_path)
    print(f"\nSaved models to:\n  - {dt_model_path}\n  - {rf_model_path}")

    # Verify reloading
    loaded_dt = joblib.load(dt_model_path)
    loaded_rf = joblib.load(rf_model_path)
    np.testing.assert_array_equal(dt_best.predict(X_test), loaded_dt.predict(X_test))
    np.testing.assert_array_equal(rf_best.predict(X_test), loaded_rf.predict(X_test))
    print("Reload verification successful! Saved model predictions match exactly.")

    # 15. Generate Markdown Report
    analysis_md_content = f"""# Decision Tree & Random Forest Technical Analysis Report

## 1. Dataset & Preprocessing Summary
- **Original Dataset Records**: {n_raw} rows, 14 columns
- **Identified Exact Duplicates**: {n_dups} rows ({n_dups/n_raw*100:.1f}%)
- **Clean Deduplicated Cohort**: {n_clean} unique patients
- **Class Distribution (Target)**:
  - Class 0 (No Heart Disease): {np.bincount(y)[0]} ({np.bincount(y)[0]/n_clean*100:.1f}%)
  - Class 1 (Heart Disease): {np.bincount(y)[1]} ({np.bincount(y)[1]/n_clean*100:.1f}%)
- **Data Leakage Prevention**: Identical duplicate rows were removed prior to train-test splitting so that repeated records do not falsely inflate test metrics.

## 2. Decision Tree Overfitting Analysis
- **Unconstrained Baseline Tree (max_depth=None)**:
  - Training Accuracy: {dt_base_train_acc:.4f} (100% memorization)
  - Testing Accuracy: {dt_base_test_acc:.4f}
  - Generalization Gap: {dt_base_train_acc - dt_base_test_acc:.4f}
- **Optimal Regularized Tree (max_depth=3)**:
  - Constraining tree depth prevents leaf nodes from isolating single training anomalies, maintaining a strong balance between bias and variance.

## 3. Hyperparameter Tuning Results
- **Decision Tree Best Parameters**:
  - Criterion: `{dt_grid.best_params_['criterion']}`
  - Max Depth: `{dt_grid.best_params_['max_depth']}`
  - Min Samples Split: `{dt_grid.best_params_['min_samples_split']}`
  - Min Samples Leaf: `{dt_grid.best_params_['min_samples_leaf']}`
  - Cross-Validation ROC-AUC: `{dt_grid.best_score_:.4f}`

- **Random Forest Best Parameters**:
  - N Estimators: `{rf_grid.best_params_['n_estimators']}`
  - Max Depth: `{rf_grid.best_params_['max_depth']}`
  - Min Samples Split: `{rf_grid.best_params_['min_samples_split']}`
  - Min Samples Leaf: `{rf_grid.best_params_['min_samples_leaf']}`
  - Max Features: `{rf_grid.best_params_['max_features']}`
  - Cross-Validation ROC-AUC: `{rf_grid.best_score_:.4f}`

## 4. Top 5 Most Important Clinical Features (Random Forest)
1. **{df_fi.iloc[0]['Feature']}** (MDI Importance: {df_fi.iloc[0]['MDI_Importance']:.4f})
2. **{df_fi.iloc[1]['Feature']}** (MDI Importance: {df_fi.iloc[1]['MDI_Importance']:.4f})
3. **{df_fi.iloc[2]['Feature']}** (MDI Importance: {df_fi.iloc[2]['MDI_Importance']:.4f})
4. **{df_fi.iloc[3]['Feature']}** (MDI Importance: {df_fi.iloc[3]['MDI_Importance']:.4f})
5. **{df_fi.iloc[4]['Feature']}** (MDI Importance: {df_fi.iloc[4]['MDI_Importance']:.4f})

## 5. 5-Fold Stratified Cross-Validation Summary
- **Tuned Decision Tree**: Mean Accuracy = {dt_cv_acc.mean():.4f} (+/- {dt_cv_acc.std():.4f}), Mean ROC-AUC = {dt_cv_roc.mean():.4f}
- **Tuned Random Forest**: Mean Accuracy = {rf_cv_acc.mean():.4f} (+/- {rf_cv_acc.std():.4f}), Mean ROC-AUC = {rf_cv_roc.mean():.4f}

## 6. Test Set Performance Summary
| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline Decision Tree | {comparison_records[0]['Test Accuracy']} | {comparison_records[0]['Precision']} | {comparison_records[0]['Recall']} | {comparison_records[0]['F1-Score']} | {comparison_records[0]['ROC-AUC']} |
| Tuned Decision Tree | {comparison_records[1]['Test Accuracy']} | {comparison_records[1]['Precision']} | {comparison_records[1]['Recall']} | {comparison_records[1]['F1-Score']} | {comparison_records[1]['ROC-AUC']} |
| Baseline Random Forest | {comparison_records[2]['Test Accuracy']} | {comparison_records[2]['Precision']} | {comparison_records[2]['Recall']} | {comparison_records[2]['F1-Score']} | {comparison_records[2]['ROC-AUC']} |
| Tuned Random Forest | {comparison_records[3]['Test Accuracy']} | {comparison_records[3]['Precision']} | {comparison_records[3]['Recall']} | {comparison_records[3]['F1-Score']} | {comparison_records[3]['ROC-AUC']} |

## 7. Educational & Clinical Disclaimer
This machine learning project is developed strictly for educational demonstration of tree-based decision models and ensemble techniques. The models have not been clinically validated and must not be used as medical diagnostic tools or substitutes for professional healthcare evaluation.
"""
    with open(os.path.join(REPORTS_DIR, 'decision_tree_analysis.md'), 'w') as f:
        f.write(analysis_md_content)
    print("Exported analysis report to outputs/reports/decision_tree_analysis.md")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    main()
