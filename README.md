# Customer Churn Prediction 


# Week 2 – Building, Evaluating, and Interpreting ML Models



## Overview

This notebook trains and evaluates several machine learning models to predict **customer churn** on the Telco Customer Churn dataset. It goes beyond accuracy by using precision, recall, F1-score, ROC-AUC, and a business-cost analysis to choose both a model and a decision threshold.

**Models compared**

- Baseline Dummy Classifier (always predicts "Stay")
- Logistic Regression
- Balanced Logistic Regression (`class_weight='balanced'`)
- Decision Tree (max depth 5)
- Random Forest (300 trees)



**Preprocessing**

1. Convert `TotalCharges` from text to numeric. The 11 blank values (all with tenure = 0) are filled with the median.
2. Drop `customerID` and the target from the features.
3. One-hot encode categorical columns (`drop_first=True`).
4. Stratified 80/20 train/test split (`random_state=42`): 5,634 training rows and 1,409 test rows.

## Notebook Structure

| Part | Topic | What it covers |
|---|---|---|
| 1 | Preprocessing, split, baseline | Cleaning, encoding, stratified split, "always stay" baseline |
| 2 | Logistic Regression | Pipeline with `StandardScaler`, odds ratios, manual sigmoid check |
| 3 | Confusion matrix and metrics | TN/FP/FN/TP, precision/recall/F1 by hand vs. scikit-learn |
| 4 | ROC-AUC, thresholds, business cost | Threshold sweep, cost-based threshold selection |
| 5 | Decision Trees and Random Forests | Depth vs. overfitting, Gini check, OOB score, permutation importance |
| 6 | Class imbalance and feature engineering | Balanced weights, engineered features (`n_services`, `is_new`, `charge_per_mo`, `price_jump`) |
| 7 | Model comparison | Final table, deployment recommendation |
| – | Check Your Understanding | Six conceptual questions with worked answers |

## Key Results

All results are on the held-out test set (1,409 customers, 374 churners).

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| LR balanced | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (d=5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | 0.842 |

### Main findings

- **Baseline:** "Always stay" scores 73.5% accuracy but catches 0 of 374 churners, so accuracy alone is misleading on this data.
- **Churn drivers (Logistic Regression):**
  - Risk factors: fiber-optic internet (odds ratio 2.18), `TotalCharges` (1.64), streaming services (about 1.29).
  - Protective factors: longer `tenure` (0.30), `MonthlyCharges` (0.40), two-year contract (0.56).
  - `TotalCharges` and `MonthlyCharges` are correlated with tenure and each other, so their individual weights should not be read as separate causes.
- **Threshold selection:** With a missed churner costing PKR 6,000 and a false alarm costing PKR 1,000, the theoretical threshold is 1000 / (1000 + 6000) ≈ 0.14. The empirical best on the 0.05 grid is **0.15**.
  - At 0.5: 212 churners caught, 110 false alarms, cost PKR 1,082,000.
  - At 0.15: 344 churners caught, 437 false alarms, cost PKR 617,000.
  - Saving: **PKR 465,000 (about 43%)**.
- **Overfitting:** Decision tree test accuracy peaks at depth 6. The unrestricted tree reaches 0.998 train vs. 0.742 test accuracy (gap 0.256).
- **Random Forest:** OOB accuracy (0.803) matches test accuracy (0.807). Top permutation-importance features are `tenure`, `TotalCharges`, and `Contract_Two year`.
- **Class weighting:** Balanced weights raise recall (0.567 to 0.781) at the cost of precision, with no change in AUC. It moves the operating point, much like lowering the threshold.
- **Feature engineering:** The engineered features did not improve the Random Forest (AUC 0.8422 to 0.8420).

### Recommendation

**Deploy Logistic Regression at threshold 0.15.** It ties the Random Forest on AUC and accuracy, achieves a lower business cost at the chosen threshold (PKR 617,000 vs. 645,000), and is easy to explain through odds ratios.

## Limitations

- The threshold was selected on the same test set used for evaluation, so the cost saving is slightly optimistic. Cross-validation (Week 3) is the proper fix.
- Cost figures (PKR 6,000 per missed churner, PKR 1,000 per unnecessary offer) are lab assumptions, not measured business data.

  ## Biggest Lesson
 
**Accuracy is not the goal. The right decision threshold, chosen using business cost, matters more than the choice of model.**
 
- **Accuracy hides failure.** A model that always predicts "Stay" scores 73.5% and catches zero churners. Logistic Regression's 80.7% still misses 162 of 374 churners (43%). Recall, precision, and AUC show what accuracy hides.
- **The model is only half the decision.** Logistic Regression and Random Forest tied on AUC (0.842), so neither separates churners better. Changing the threshold from 0.5 to 0.15 raised recall from 0.567 to 0.920 and cut the cost by PKR 465,000 (about 43%), far more than switching models did.
- **Costs are not symmetric.** A missed churner (PKR 6,000) costs six times a false alarm (PKR 1,000). Decision theory gives the threshold directly: t* = C_FP / (C_FP + C_FN) ≈ 0.14, which the empirical search confirmed.
- **Fancier is not always better.** The Random Forest, balanced class weights, and engineered features did not improve AUC. The simple, explainable Logistic Regression was the best choice to deploy.
- The threshold was selected on the same test set used for evaluation, so the cost saving is slightly optimistic. Cross-validation (Week 3) is the proper fix.
- Cost figures (PKR 6,000 per missed churner, PKR 1,000 per unnecessary offer) are lab assumptions, not measured business data.



```bash
```

# Week 3: Model Optimization and Unsupervised Learning

**Author:** Raja Fahad kiani
**Course:** Introduction to Applied AI
**Date:** 3 October, 2026
**Dataset:** Telco Customer Churn (7,043 customers, 30 features after one-hot encoding; 5,634 train / 1,409 test)

## Results Summary

- **Split-to-split accuracy range across 20 seeds:** 0.780 to 0.828 (std 0.0104; theoretical 95% CI is about ±0.021, which matches the observed spread)
- **5-fold CV AUC (tuned models):**
  - LR: 0.846 ± 0.013
  - RF: 0.846 ± 0.011
  - XGBoost: 0.850 ± 0.012
- **Tuning:**
  - Best RF params (random search, used in the final comparison): `max_depth=15`, `max_features≈0.213`, `min_samples_leaf=15` (CV AUC 0.8464)
  - Best RF params (grid search): `max_depth=8`, `max_features='sqrt'`, `min_samples_leaf=20` (CV AUC 0.8468)
  - Grid vs random search time: 105 s (24 combos, 120 fits) vs 108 s (24 iterations, 120 fits). The AUCs differ by only 0.0004, so the two methods performed the same.
  - Best XGBoost params: `max_depth=2`, `learning_rate≈0.034`, `n_estimators=476`, `subsample=0.6`, `colsample_bytree≈0.561`, `min_child_weight=1`, `reg_lambda≈1.97`
- **Test AUC of final model (used once):** 0.8483 (tuned XGBoost; recall 0.521, precision 0.659 at a 0.5 threshold)
- **Customer segments (k = 4):**
  - **New, low-spend, few services:** 1,918 customers, 32% churn
  - **Mid-tenure, high-spend, few add-ons:** 2,157 customers, 43% churn (highest risk)
  - **Loyal, high-spend, heavy bundle:** 1,938 customers, 14% churn
  - **Loyal, low-spend, basic plan:** 1,030 customers, 5% churn (lowest risk)
- **PCA:** 15 of 30 components explain 90% of the variance
- **Biggest lesson:** The gaps between tuned models (about 0.004 AUC) are smaller than the fold-to-fold noise (about 0.012) and the split-to-split noise (about ±2 points of accuracy), so a single split or a small score difference should never be treated as proof that one model is better.

## Segment Profiles

| Segment | Customers | Avg tenure (months) | Avg monthly ($) | Avg services | Churn |
|---|---|---|---|---|---|
| Mid-tenure, high-spend, few add-ons | 2,157 | 18.4 | 80.41 | 3.3 | 43% |
| New, low-spend, few services | 1,918 | 9.0 | 37.71 | 1.2 | 32% |
| Loyal, high-spend, heavy bundle | 1,938 | 59.8 | 92.09 | 5.1 | 14% |
| Loyal, low-spend, basic plan | 1,030 | 53.6 | 30.96 | 1.5 | 5% |

Churn was not used to form the clusters, only to profile them.

## Method Notes

1. **Split variance:** Logistic Regression was retrained on 20 random 75/25 splits of the training data to show how much accuracy moves from the split alone.
2. **Model comparison:** Stratified 5-fold CV, with the scaler inside a `Pipeline` to avoid leakage. The untuned baseline gave LR 0.846 and RF 0.844 AUC.
3. **Tuning:** LR was tuned with a validation curve (best C = 10). RF was tuned with both grid and random search. XGBoost used early stopping (best at 247 trees, validation AUC 0.854) and then random search over 30 iterations.
4. **Model selection:** Tuned XGBoost had the best CV AUC and was evaluated on the locked test set exactly once, then saved as `churn_model.joblib` for Week 4.
5. **Segmentation:** K-means on standardized tenure, MonthlyCharges, TotalCharges, and number of services, with k = 4 chosen from the elbow and silhouette plots.
6. **PCA:** The first principal component is dominated by the "No internet service" dummy columns, which have identical loadings (0.302). These columns are redundant copies of the same information, which is why 30 features compress to fewer components.

## Caveats

- Test AUC (0.848) is close to CV AUC (0.850), which suggests no serious overfitting to the CV folds.
- Recall of 0.52 means about half of churners are missed at the default 0.5 threshold. A lower threshold may be worth exploring in Week 4.
- The segmentation was fit on all 7,043 customers, not only the training set. This is acceptable for unsupervised profiling but should not be used to build features for the predictive model without refitting inside CV.





**Biggest lesson:** A small score gap between models, like the 0.004 AUC between tuned XGBoost and the others, is smaller than the noise from CV folds (about 0.012) and random splits (about ±2 points of accuracy), so I can't treat a single split or a tiny difference as proof that one model is better.
