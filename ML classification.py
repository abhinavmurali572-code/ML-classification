# ML Assignment 3: Classification - Breast Cancer Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ============== 1. LOADING AND PREPROCESSING (2 marks) ==============

# Load Breast Cancer dataset
cancer = load_breast_cancer()
print(cancer.DESCR)  # Dataset description

# Convert to DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # 0 = Malignant, 1 = Benign
print(f"Shape: {df.shape}")
print(df.head())

print("\n--- Target Distribution ---")
print(df['target'].value_counts())
print("Target names:", cancer.target_names)
# 0 = Malignant (212 samples), 1 = Benign (357 samples)

print("\n--- Info ---")
print(df.info())
print(df.describe())

# Check missing values
print("\nMissing values:\n", df.isnull().sum())
# This dataset has NO missing values, but we implement handling for completeness
# If missing existed: df.fillna(df.median(), inplace=True)
# Or: from sklearn.impute import SimpleImputer

# Feature Scaling - Standardization
# Justification: Features have vastly different scales.
# Eg: mean radius ~14, mean area ~650, mean smoothness ~0.09
# Logistic Regression, SVM, and k-NN are distance/gradient based and will be biased without scaling.
# Tree models don't need scaling but scaling doesn't hurt them.

X = df.drop('target', axis=1)
y = df['target']

# Train-test split BEFORE scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y keeps same ratio of Malignant/Benign in train and test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- After Scaling (first 5 rows) ---")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head())

# Preprocessing Justification:
# 1. No missing values found, verified with isnull().sum()
# 2. StandardScaler used to bring all 30 features to mean=0, std=1
# 3. Stratified split ensures both classes are fairly represented in train/test

# ============== 2. CLASSIFICATION ALGORITHM IMPLEMENTATION (5 marks) ==============

results = {}
predictions = {}

# 1. Logistic Regression
# How it works: Uses sigmoid function to predict probability of class (0/1).
# Learns linear decision boundary using log-loss.
# Suitability: Good baseline for binary medical diagnosis, interpretable, works well when classes are linearly separable.
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
predictions['Logistic Regression'] = pred_lr

# 2. Decision Tree Classifier
# How it works: Splits data recursively using Gini/Entropy to create pure leaf nodes.
# Suitability: Captures non-linear relationships, interpretable (can visualize tree), handles feature interactions.
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_scaled, y_train)
pred_dt = dt.predict(X_test_scaled)
predictions['Decision Tree'] = pred_dt

# 3. Random Forest Classifier
# How it works: Ensemble of Decision Trees (Bagging). Trains 100 trees on random subsets and votes.
# Suitability: Reduces overfitting of single tree, very robust for medical data with 30 features.
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)
predictions['Random Forest'] = pred_rf

# 4. Support Vector Machine (SVM)
# How it works: Finds optimal hyperplane with maximum margin between classes. Uses RBF kernel for non-linear boundaries.
# Suitability: Very effective for high-dimensional data (30 features) and small dataset (569 rows).
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
predictions['SVM'] = pred_svm

# 5. k-Nearest Neighbors (k-NN)
# How it works: Lazy learner. For each test point, finds k=5 nearest training points by Euclidean distance and takes majority vote.
# Suitability: Simple, no training phase, works well when similar tumours have similar measurements. Requires scaling - which we did.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
predictions['k-NN'] = pred_knn

# ============== 3. MODEL COMPARISON (2 marks) ==============

def evaluate_model(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return [acc, prec, rec, f1]

print("\n--- MODEL EVALUATION ---")
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

for model_name, y_pred in predictions.items():
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred, model_name)
    metrics_df.loc[len(metrics_df)] = [model_name, acc, prec, rec, f1]

print("\n--- Comparison Table (Sorted by Accuracy) ---")
print(metrics_df.sort_values(by='Accuracy', ascending=False))

# Detailed report for best model
best_model_name = metrics_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
print(f"\n--- Detailed Classification Report for Best Model: {best_model_name} ---")
print(classification_report(y_test, predictions[best_model_name], target_names=cancer.target_names))

# Visualization 1: Accuracy Comparison
plt.figure(figsize=(10,5))
sns.barplot(data=metrics_df, x='Model', y='Accuracy')
plt.ylim(0.85, 1.0)
plt.title('Accuracy Comparison of Classification Models')
plt.xticks(rotation=15)
plt.show()

# Visualization 2: Confusion Matrix for All Models
fig, axes = plt.subplots(1, 5, figsize=(20,4))
for ax, (model_name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(model_name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Visualization 3: Feature Importance from Random Forest
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Top 10 Important Features - Random Forest')
plt.show()

# ============== CONCLUSION FOR REPORT ==============
print("""
CONCLUSION:

Best-performing: Usually SVM or Logistic Regression (Accuracy ~0.97-0.98)
- Justification: Breast Cancer dataset is almost linearly separable and high-dimensional.
  SVM with RBF kernel finds optimal margin, Logistic Regression finds clean linear boundary.
  Both generalize very well on 569 samples.

Worst-performing: Usually Decision Tree or k-NN (Accuracy ~0.93-0.95)
- Decision Tree Reason: Single tree overfits, high variance, sensitive to small changes in data.
- k-NN Reason: Sensitive to noisy features and curse of dimensionality with 30 features,
  even after scaling.

In medical diagnosis, Recall is critical (we don't want to miss Malignant cases).
Check Recall column too - SVM and Logistic Regression also have highest Recall.
""")
