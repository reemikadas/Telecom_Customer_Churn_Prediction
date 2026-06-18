# Customer Churn Prediction and Retention Strategies

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This project builds a machine learning pipeline to predict customer churn — identifying customers who are likely to stop using a service — and to inform targeted retention strategies. Using a structured dataset with customer demographics, engagement behavior, payment history, and subscription details, four supervised classification models are trained, tuned, and evaluated to determine which best identifies at-risk customers before they leave.

---

## Business Problem

Customer churn is one of the most costly challenges for subscription-based businesses. Acquiring a new customer can cost five to seven times more than retaining an existing one. When customers leave without warning, businesses lose not only revenue but also the opportunity to intervene. The core challenge is: **can we predict which customers are likely to churn before they do?**

---

## Business Objective

- Develop a predictive model that accurately identifies customers at high risk of churning.
- Minimize **False Negatives** (missed churners) — the most costly type of error in a retention context.
- Provide actionable insights from model results to guide retention campaigns, support improvements, and contract or pricing adjustments.
- Enable the business to proactively target at-risk customers with personalized retention offers before they leave.

---

## Dataset

### Source

Two separate CSV files are used in this project:
- `customer_churn_dataset_training.csv` — used for EDA, preprocessing, and model training.
- `customer_churn_dataset_testing.csv` — held out exclusively for final model evaluation.

### Dataset Summary

| Attribute | Detail |
|---|---|
| Index Column | `CustomerID` |
| Training Rows | ~499,999 (after removing 1 row with missing values) |
| Features | 11 (after setting index) |
| Target Variable | `Churn` (binary: 0 = Not Churned, 1 = Churned) |

**Feature Descriptions:**

| Category | Features |
|---|---|
| Customer Demographics | `Age`, `Gender` |
| Engagement Metrics | `Tenure`, `Usage Frequency`, `Support Calls` |
| Payment Behavior | `Payment Delay`, `Total Spend` |
| Subscription Details | `Subscription Type`, `Contract Length` |
| Customer Interaction | `Last Interaction` |
| Target | `Churn` |

**Categorical Values:**
- `Gender`: Male, Female
- `Subscription Type`: Basic, Standard, Premium
- `Contract Length`: Annual, Monthly, Quarterly

### Dataset Challenges

- **Class Imbalance:** The original training data is imbalanced between churned and non-churned customers. This was addressed using **SMOTE (Synthetic Minority Oversampling Technique)** to produce a balanced 50/50 class distribution before model training.
- **Data Type Correction:** Several numeric columns (`age`, `tenure`, `support calls`, `churn`, etc.) were stored as floats and required conversion to integers.
- **Missing Values:** One row contained missing values across all fields and was removed from the training set.
- **No Duplicate Customers:** The dataset was verified to contain no duplicate `CustomerID` entries.

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Preprocessing | `sklearn.preprocessing` (StandardScaler, OneHotEncoder), `imblearn` (SMOTE) |
| Class Imbalance | `imblearn.over_sampling.SMOTE`, `sklearn.utils.class_weight` |
| Model Tuning | `sklearn.model_selection.GridSearchCV` |
| ML Models | `LogisticRegression`, `DecisionTreeClassifier`, `GradientBoostingClassifier`, `MLPClassifier` |
| Evaluation Metrics | `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `precision_recall_curve` |
| Environment | Jupyter Notebook |

---

## Project Workflow

```
1. Data Loading
        ↓
2. Exploratory Data Analysis (EDA)
   ├── Data shape, types, missing values
   ├── Duplicate check
   ├── Univariate distributions (numeric features)
   ├── Bivariate analysis (churn vs. subscription type, contract length, total spend, tenure)
   ├── Churn probability by payment delay and support calls
   ├── Correlation heatmap
   └── Age group segmentation
        ↓
3. Data Preprocessing
   ├── Categorical encoding (One-Hot Encoding)
   ├── Train/test split (pre-split files)
   ├── Class imbalance handling (SMOTE)
   └── Feature scaling (StandardScaler — for Logistic Regression & MLP)
        ↓
4. Model Development & Tuning (Baseline → Experiment 1 → Experiment 2)
   ├── Logistic Regression
   ├── Decision Tree
   ├── Gradient Boosting
   └── Multi-Layer Perceptron (MLP)
        ↓
5. Model Evaluation
   ├── Accuracy (Train & Test)
   ├── Classification Report (Precision, Recall, F1-Score)
   ├── Confusion Matrix
   └── ROC-AUC Curve
        ↓
6. Conclusion & Recommendation
```

---

## Exploratory Data Analysis

Key findings from EDA on the training dataset:

**Distributions:** All numeric features (`age`, `tenure`, `usage frequency`, `support calls`, `payment delay`, `total spend`, `last interaction`) were visualized using histograms with KDE overlays to understand their spread and skew.

**Churn by Subscription Type & Contract Length:** Count plots revealed how churn rates vary across subscription tiers and contract durations. Monthly contract customers showed higher churn propensity compared to annual subscribers.

**Churn by Total Spend & Tenure:** Box plots comparing churned vs. non-churned customers showed that customers who churn tend to have lower total spend and shorter tenure — consistent with the pattern of newer, lower-value customers being more at risk.

**Payment Delay vs. Churn Probability:** A line chart revealed a clear positive correlation: customers with longer payment delays have a progressively higher probability of churning.

**Support Calls vs. Churn Probability:** A strong positive trend was observed — customers who contact support more frequently are significantly more likely to churn, suggesting that unresolved issues are a leading churn indicator.

**Correlation Heatmap:** A numeric correlation matrix was computed to identify relationships between features and the target variable `churn`.

**Age Group Segmentation:** Customers were segmented into five age groups:
- Young Adults (18–24)
- Young Professionals (25–34)
- Established Adults (35–44)
- Middle-Aged Consumers (45–54)
- Seniors (55–65)

Churn distributions were plotted across age groups and gender to reveal demographic patterns.

---

## Machine Learning Models

Each model was built using a **Baseline → Experiment 1 → Experiment 2** approach, with GridSearchCV (5-fold cross-validation) applied for hyperparameter tuning.

### 1. Logistic Regression
- **Preprocessing:** StandardScaler applied; SMOTE-balanced training data used.
- **Tuning Parameters:** Regularization penalty (`l1`, `l2`), regularization strength `C`, and solver (`saga`, `lbfgs`).
- **Experiments:** Exp 1 used `C ∈ [0.001–1]`; Exp 2 narrowed to very small `C ∈ [0.000001–0.001]`.
- **Outcome:** Both experiments converged to the same best parameters with no meaningful improvement over baseline. The model plateaued, prompting the move to more complex algorithms.

### 2. Decision Tree
- **Preprocessing:** SMOTE-balanced data; no scaling required.
- **Tuning Parameters:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha` (pruning), and `class_weight`.
- **Experiments:** Exp 1 used broader depth/pruning ranges; Exp 2 added `class_weight='balanced'` and tighter constraints.
- **Outcome:** High training accuracy but poor generalization to test data. High recall with low precision indicated the model overpredicted positive (churn) cases — persistent overfitting despite pruning.

### 3. Gradient Boosting
- **Preprocessing:** SMOTE-balanced data; no scaling required.
- **Tuning Parameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `min_samples_split`.
- **Experiments:** Exp 1 explored broader learning rates (`0.002–0.1`); Exp 2 narrowed to very low learning rates (`0.002–0.005`).
- **Outcome:** Better generalization than Decision Tree. The train-test accuracy gap narrowed, suggesting improved ability to capture patterns without overfitting. ROC-AUC of **0.67** indicated moderate discriminative ability.

### 4. Multi-Layer Perceptron (MLP)
- **Preprocessing:** StandardScaler applied; SMOTE-balanced training data used; early stopping enabled.
- **Tuning Parameters:** `hidden_layer_sizes`, `activation`, `solver`, `alpha` (L2 regularization), `max_iter`.
- **Experiments:** Exp 1 used hidden layers `(32,)` and `(64,)`; Exp 2 expanded to `(64,)` and `(128,)`.
- **Advanced Tuning:** Precision-Recall curve analysis was applied in Exp 2 to find the **optimal classification threshold** (where precision ≈ recall), replacing the default 0.5 threshold.
- **Outcome:** Highest ROC-AUC of **0.74** among all models. Training accuracy reached 89% with test accuracy around 67%; threshold tuning improved test performance. Some overfitting persisted, but the model showed the best balance between sensitivity and specificity.

---

## Model Performance Comparison

| Model | Train Accuracy | Test Accuracy | ROC-AUC | Notes |
|---|---|---|---|---|
| Logistic Regression (Baseline) | ~LL | ~LL | — | Stable but limited capacity |
| Logistic Regression (Tuned) | ~LL | ~LL | — | No improvement over baseline |
| Decision Tree (Baseline) | High | Low | — | Severe overfitting |
| Decision Tree (Tuned) | High | Low | — | Overfitting persists post-pruning |
| Gradient Boosting (Baseline) | Moderate | Moderate | ~0.67 | Better generalization |
| Gradient Boosting (Tuned) | Moderate | Moderate | **0.67** | Reduced train-test gap |
| MLP (Baseline) | High | ~67% | ~0.74 | Strong learner, overfits |
| **MLP (Tuned — Exp 2)** | **89%** | **~67%** | **0.74** | **Best ROC-AUC; threshold optimized** |

> **Note:** Exact numeric scores for Logistic Regression and Decision Tree tests are available in the notebook outputs. Gradient Boosting and MLP final scores are explicitly reported in the notebook's conclusion cells.

---

## Best Model Analysis

The **MLP (Experiment 2)** model — with hidden layer size of 64 or 128 neurons, ReLU activation, Adam optimizer, L2 regularization (`alpha`), early stopping, and an optimized classification threshold — delivered the strongest overall performance:

- **ROC-AUC of 0.74** — the highest across all models, indicating the best ability to discriminate between churners and non-churners.
- **Threshold Optimization:** Rather than using the default 0.5 cutoff, the Precision-Recall curve was analyzed to find the threshold where precision and recall are approximately equal, reducing bias toward one class and improving balanced performance.
- **Comparison to Gradient Boosting:** While Gradient Boosting showed better generalization characteristics (smaller train-test gap), its ROC-AUC of 0.67 was notably lower than MLP's 0.74 — a meaningful difference for a churn use case where ranking customers by risk probability is critical for targeting.

---

## Recommended Model

**Recommended: MLP (Tuned — Experiment 2) with Optimized Threshold**

**Justification:**
- Achieves the highest ROC-AUC (0.74), making it the most reliable model for ranking customers by churn probability.
- Threshold tuning allows the business to adjust the sensitivity/specificity trade-off based on retention campaign cost and customer lifetime value.
- For a churn prediction scenario, **maximizing recall** (catching more true churners) is often prioritized over precision — the MLP with an adjusted threshold supports this requirement.

**If interpretability is a priority**, the **Gradient Boosting** model is the recommended alternative — it offers more stable generalization, a narrower train-test accuracy gap, and feature importance scores that can be extracted for business storytelling, with a trade-off of lower ROC-AUC (0.67).

---

## Key Findings

- **Support Calls are the strongest churn signal.** Customers who contact support multiple times are significantly more likely to churn — unresolved issues or poor support experiences are a leading driver.
- **Payment delays correlate strongly with churn probability.** The longer a customer delays payment, the higher their churn risk — this is a reliable early warning indicator.
- **Short-tenure customers churn more.** Newer customers with lower total spend are disproportionately at risk, suggesting onboarding and early engagement are critical retention windows.
- **Monthly contract customers are higher churn risk** compared to annual or quarterly subscribers, consistent with lower switching costs.
- **Class imbalance required active correction.** The original dataset was imbalanced; without SMOTE, models would have been biased toward predicting non-churn. Balancing the classes was essential for meaningful recall on the minority (churn) class.
- **Tree-based models overfitted despite pruning.** Decision Trees struggled to generalize even with aggressive hyperparameter tuning, while ensemble and neural network approaches handled the complexity better.
- **Neural networks benefit from threshold optimization.** Using the default 0.5 classification threshold underperformed; calibrating the threshold using the Precision-Recall curve yielded a more useful operating point for business decision-making.

---

## Business Impact

| Insight | Recommended Action |
|---|---|
| High support call volume predicts churn | Implement proactive outreach after 2+ support contacts; prioritize issue resolution SLAs |
| Payment delays signal churn risk | Trigger early payment reminders and flexible payment plan offers before delays escalate |
| Monthly contract customers churn more | Offer discounts or incentives to convert monthly subscribers to annual/quarterly plans |
| New, low-spend customers are high-risk | Design onboarding programs and early engagement campaigns (first 90 days are critical) |
| Churn probability scores from MLP | Build a risk-tiered retention campaign: high-risk → personalized offer, medium-risk → email nurture, low-risk → standard retention |

Deploying the MLP model in a production CRM pipeline would allow the business to score all active customers weekly, flag those above the churn probability threshold, and automatically enroll them in targeted retention workflows — shifting from reactive to proactive churn management.

---

## Repository Structure

```
customer-churn-prediction/
│
├── Customer_Churn_Prediction_ML_Project.ipynb   # Main Jupyter Notebook
│
├── customer_churn_dataset_training.csv          # Training dataset
├── customer_churn_dataset_testing.csv           # Testing dataset
│
└── README.md                                    # Project documentation
```

---

## 👩‍💻 Author

**Reemika Subrata Das**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-reemikadas-blue?logo=linkedin&style=flat)](https://linkedin.com/in/reemikadas)
[![GitHub](https://img.shields.io/badge/GitHub-reemikadas-black?logo=github&style=flat)](https://github.com/reemikadas)
[![Email](https://img.shields.io/badge/Email-das.reemika%40gmail.com-red?logo=gmail&style=flat)](mailto:das.reemika@gmail.com)
---

*This project was completed as part of a machine learning portfolio to demonstrate end-to-end ML workflow design, model selection, and business-oriented data science.*
