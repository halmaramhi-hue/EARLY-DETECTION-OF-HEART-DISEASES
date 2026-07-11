# 🫀 Early Detection of Heart Diseases — Solution #3: K-Means + SVM (Hybrid)

**Course:** CCAI323 · Machine Learning — Final Project
**Team:** More Beat
**My Contribution:** Solution #3 — Hybrid K-Means + SVM Classifier

---

## 📋 Project Context
This is a comparative study by Team More Beat evaluating **four ML paradigms** plus a **state-of-the-art benchmark** for predicting heart disease from the **Cleveland Heart Disease dataset (UCI)** — 297 patients, 13 clinical features, no missing values.

| # | Solution | Author |
|---|----------|--------|
| 1 | Neural Network (MLP) | Emad Alzahrani |
| 2 | Logistic Regression | Hani Alkhmesy |
| **3** | **K-Means + SVM (Hybrid)** | **Hashim Almaramihi** ← *my work* |
| 4 | Gaussian Naïve Bayes | Mohammed Almutari |
| SOA | Random Forest (replicated) | Ahmed Bawazeer |

---

## 🎯 My Approach: Hybrid Clustering + Classification
Instead of feeding raw features directly into a classifier, this solution uses **K-Means as a feature-engineering step**: it clusters patients first, then feeds the SVM extra "distance-to-centroid" features alongside the original ones — enriching the input space with structural information the SVM alone wouldn't see.

**Pipeline:**
1. Split & scale data
2. Find optimal `k` using silhouette score
3. Fit K-Means → extract centroid distances as new features
4. Concatenate original + centroid-distance features
5. Tune an SVM (`GridSearchCV`) on the enriched feature set
6. Compare against a baseline SVM (no clustering) to isolate K-Means' contribution

---

## 🛠️ Libraries Used
| Library | Purpose |
|---------|---------|
| `pandas` | Loading and exploring the dataset |
| `numpy` | Array operations, stacking hybrid features |
| `matplotlib` | Silhouette score plot, confusion matrix visualization |
| `scikit-learn` (`sklearn`) | |
| &nbsp;&nbsp;`train_test_split`, `StratifiedKFold` | Stratified 80/20 split preserving class balance |
| &nbsp;&nbsp;`StandardScaler` | Feature scaling (required for both K-Means and SVM) |
| &nbsp;&nbsp;`KMeans` | Unsupervised clustering / feature engineering |
| &nbsp;&nbsp;`SVC` | Support Vector Machine classifier |
| &nbsp;&nbsp;`GridSearchCV` | Hyperparameter tuning (`C`, `gamma`, `kernel`) |
| &nbsp;&nbsp;`silhouette_score` | Selecting the optimal number of clusters |
| &nbsp;&nbsp;`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` | Model evaluation |
| &nbsp;&nbsp;`classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay` | Detailed performance reporting |

---

## 📊 Results

| Metric | Baseline SVM | **Hybrid K-Means + SVM** |
|--------|:---:|:---:|
| Accuracy | 84.2%* | **91.67%** |
| Precision | — | **100.00%** |
| Recall | ~71%* | **82.14%** |
| F1-score | — | **90.20%** |
| ROC-AUC | — | **0.9319** |

*\*Baseline SVM values from the notebook's internal comparison (recall lift of ~11pp mentioned in the presentation).*

**Confusion Matrix (test set, n=60):**
```
                Pred: No   Pred: Disease
Actual: No         32           0
Actual: Disease      5          23
```

### Key Finding
The centroid-distance features improved **recall from 71% → 82%** without sacrificing precision (still 100%, zero false positives). This solution achieved the **best overall Accuracy and F1-score** among all 5 models compared in the project.

---

## 📁 Files
- `Project_ML__2_.ipynb` — full notebook (data loading → clustering → hybrid feature construction → SVM tuning → evaluation → baseline comparison)
- `More_Beat_Final_Presentation.pdf` — full team presentation (all 5 solutions + comparative analysis)

---

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
```
```python
# Place heart_cleveland_upload.csv in the working directory, then run the notebook top to bottom.
```

---

## 🔑 Takeaway
> Hybridization only adds value when it introduces **non-redundant signal**. Here, K-Means' centroid distances captured structural patient similarity that raw features alone didn't expose to the SVM — directly boosting recall, which matters most in a clinical screening context where missed diagnoses are costlier than false alarms.
