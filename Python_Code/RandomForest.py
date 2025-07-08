# -*- coding: utf-8 -*-
"""
Created on Fri May 23 09:56:58 2025

@author: vince
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import string

# -----------------------------
# Data Preparation
# -----------------------------

def load_and_clean_data(path: str, filename: str) -> pd.DataFrame:
    """
    Load CSV data, convert light_class to numeric and remove NaNs.

    Args:
        path (str): Path to the CSV file.
        filename (str): File name of the CSV.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    data = pd.read_csv(f"{path}/{filename}")
    data['light_class'] = pd.to_numeric(data['light_class'], errors='coerce')
    return data.dropna(subset=['light_class'])

def standardize_features(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Standardize numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        exclude_columns (list): Columns to exclude from standardization.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    features = df.drop(columns=exclude_columns)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# -----------------------------
# Feature Selection
# -----------------------------

def forward_feature_selection(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv) -> list:
    """
    Perform manual forward feature selection.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        groups (pd.Series): Group labels for cross-validation.
        cv: Cross-validation splitter.

    Returns:
        list: Selected feature names.
    """
    selected, remaining = [], list(X.columns)
    best_score = 0

    print("Starting feature forward selection...")
    while remaining:
        scores = []
        for feat in remaining:
            current = selected + [feat]
            score = cross_val_score(RandomForestClassifier(max_features='sqrt', random_state=42),
                                    X[current], y, groups=groups, cv=cv).mean()
            scores.append((score, feat))

        scores.sort(reverse=True)
        if scores[0][0] > best_score:
            best_score, best_feat = scores[0]
            selected.append(best_feat)
            remaining.remove(best_feat)
            print(f"Selected: {selected}, Accuracy: {best_score:.4f}")
        else:
            break
    return selected

# -----------------------------
# Model Training and Evaluation
# -----------------------------

def perform_grid_search(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv):
    """
    Run GridSearchCV to find best hyperparameters.

    Returns:
        GridSearchCV: Fitted GridSearchCV object.
    """
    param_grid = {"max_features": [1, 2, 3, 4, 5, "sqrt", "log2"]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    return grid_search

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv):
    """
    Evaluate model using cross_val_predict and show confusion matrix.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, 
                         index=[f"True_{cls}" for cls in sorted(y.unique())],
                         columns=[f"Pred_{cls}" for cls in sorted(y.unique())])

    print("Out-of-Fold Confusion Matrix:\n", cm_df)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Out-of-Fold)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Statistical Analysis
# -----------------------------

def get_cld(pval_df: pd.DataFrame, means: pd.Series, alpha: float = 0.05) -> dict:
    """
    Generate compact letter display for posthoc test results.

    Args:
        pval_df (pd.DataFrame): DataFrame with p-values.
        means (pd.Series): Means of the groups.
        alpha (float): Significance level.

    Returns:
        dict: Mapping from group to letter.
    """
    sorted_groups = means.sort_values().index.tolist()
    letters = {group: '' for group in sorted_groups}
    current_letter = 'a'
    assigned = set()

    for group in sorted_groups:
        if group in assigned:
            continue
        letters[group] += current_letter
        assigned.add(group)
        for other in sorted_groups:
            if other not in assigned and pval_df.loc[group, other] > alpha:
                letters[other] += current_letter
                assigned.add(other)
        current_letter = chr(ord(current_letter) + 1)

    return letters

def plot_metrics_by_class(df: pd.DataFrame, metrics: list, class_col: str = 'light_classes_int'):
    """
    Generate boxplots with Tukey-HSD posthoc and CLD for each metric.
    """
    df[class_col] = np.floor(df["light_class"] + 0.5).astype(int)

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=class_col, y=metric, data=df)
        plt.title(f'({string.ascii_uppercase[i]}) {metric}')
        plt.xlabel(class_col)
        plt.ylabel(metric)

        posthoc = sp.posthoc_tukey(df, val_col=metric, group_col=class_col)
        means = df.groupby(class_col)[metric].mean()
        cld = get_cld(posthoc, means)

        ymax = df[metric].max()
        for x, group in enumerate(sorted(df[class_col].unique())):
            plt.text(x, ymax * 1.05, cld[group], ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------

def pipeline(file_name):
    # Load and prepare data
    path = 'Data'
    df = load_and_clean_data(path, file_name)
    y = np.floor(df["light_class"] + 0.5).astype(int)
    groups = df["species"]
    X_scaled = standardize_features(df, exclude_columns=["Unnamed: 0", "species", "light_class"])

    # Feature selection
    gkf = GroupKFold(n_splits=5)
    selected_features = forward_feature_selection(X_scaled, y, groups, gkf)

    # Hyperparameter tuning
    grid = perform_grid_search(X_scaled[selected_features], y, groups, gkf)
    print("\nBest parameters (mtry):", grid.best_params_)
    print("Best accuracy:", grid.best_score_)

    # Evaluation
    evaluate_model(grid.best_estimator_, X_scaled[selected_features], y, groups, gkf)

    # Statistical analysis and visualization
    metrics = ['centroid_sun', 'centroid_diff', 'peak_density', 'peak_x', 'peak_y',
               'peak_centroid_distance', 'area_68', 'area_95', 'width_68', 'height_68',
               'aspect_ratio_68', 'area_ratio_68_95']
    plot_metrics_by_class(df, metrics)

if __name__ == "__main__":
    pipeline('light_properties_Somme.csv')
    pipeline('light_properties_Lozere.csv')
    pipeline('energy_properties_Somme.csv')
    pipeline('energy_properties_Lozere.csv')
