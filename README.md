# Bankruptcy Prediction Using Machine Learning

Predicts the probability that a company will go bankrupt, using financial ratio data and several classic ML models.

## Dataset

- Source: [Company Bankruptcy Prediction (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
- Taiwan Economic Journal data, 1999–2009. Bankruptcy is defined per Taiwan Stock Exchange business regulations.
- 95 financial ratio columns per company; target is bankruptcy status.

## Approach

1. **Load & clean data** — import with pandas, check for missing values and duplicates.
2. **Feature selection** — `SelectKBest` (ANOVA F-value) picks the top 25 features.
3. **Feature reduction** — drop highly correlated features (correlation > 0.85), leaving 13 final features.
4. **Handle class imbalance** — oversample the minority class with SMOTE.
5. **Train models** — Decision Tree, Random Forest, Logistic Regression, Naive Bayes, SVM, and XGBoost, each tuned with Bayesian optimization (`BayesSearchCV`).
6. **Ensemble** — combine all models into a soft-voting `VotingClassifier` for the final prediction.
7. **Evaluate** — accuracy, precision, recall, F1, AUC-ROC, and confusion matrices on both balanced and unbalanced data.

## Key Financial Ratios Used

Net income/total assets, net worth/assets, retained earnings/total assets, EPS, current liabilities/assets, working capital/total assets, current ratio, ROE, net asset value per share, liabilities-assets flag, total expense/assets, borrowing dependency, working capital/equity.

## Results

The Voting Classifier ensemble outperformed the individual models across accuracy, precision, recall, F1, and AUC-ROC. Full metrics and plots are in the notebook; a summary table is written to `output.csv`.

## Requirements

- Python 3
- pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), xgboost, scikit-optimize (BayesSearchCV), matplotlib/seaborn (for plots)

## Usage

1. Download the dataset from Kaggle and place the CSV in the project directory.
2. Run the notebook/script end-to-end (data loading → feature selection → model training → evaluation).
3. Check `output.csv` for the model comparison summary.
