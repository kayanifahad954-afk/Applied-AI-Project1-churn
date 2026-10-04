# Customer Churn Prediction 

# Week 3: Model Optimization and Unsupervised Learning

**Author:** Syed Najeeb Ullah Shah
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
