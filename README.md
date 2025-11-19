Credit Risk Scoring: Predicting Loan Defaults

Business Context:
In the lending industry, financial institutions face a dual challenge:
  Too Conservative: Loss of revenue (opportunity cost).
  Too Aggressive: Loss of capital (default risk).
This project simulates a real-world Production Data Science workflow. My goal was not simply to "train a model," but to build a reliable risk assessment system that accurately ranks borrowers from "safe" to "risky."
  Key Highlight: The project prioritizes Domain-Driven Feature Engineering (creating synthetic financial ratios) over blind hyperparameter tuning, reflecting a deep understanding of the underlying business logic.

Performance:
The model was evaluated using ROC-AUC, as the primary business objective is the correct ranking of customers rather than hard classification.
Model	ROC-AUC (OOF)	Role in Pipeline
LightGBM	0.9210	Best single model (Speed/Accuracy balance)
XGBoost	0.9206	Adds diversity to the ensemble
CatBoost	0.9191	Excellent handling of categorical features
Ensemble (Blend)	0.9208	Final Output (Weighted Average + Rank)

Tech Stack:
Language: Python 3
Data Processing: Pandas, NumPy, SciPy (Skewness/Kurtosis analysis)
Modeling: LightGBM, XGBoost, CatBoost
Validation: Scikit-learn (Stratified K-Fold)
Visualization: Matplotlib, Seaborn

1. Deep EDA & Data Cleaning
Analyzed Class Imbalance to determine the need for stratification.
Implemented Outlier Detection using the IQR (Interquartile Range) method.
Investigated multicollinearity to reduce noise.
2. Advanced Feature Engineering (The Core Value)
Instead of relying on raw data, I engineered 70+ new features based on financial domain knowledge:
Financial Ratios: Derived Debt-to-Income Ratio, Disposable Income, and Credit Utilization metrics.
Risk Scoring: Manually constructed composite scores (Risk Score v1-v3) combining credit history and debt burden.
Interaction Features: Created polynomial interactions to help tree-based models detect non-linear patterns that simple splits might miss.
3. Robust Validation Strategy
Utilized Stratified K-Fold Cross-Validation (5 folds).
This ensures the target distribution remains consistent across all folds, preventing Data Leakage and providing a realistic estimate of how the model will perform on new, unseen clients.
4. Advanced Ensembling
Implemented multiple blending strategies to squeeze maximum performance:
Weighted Averaging (based on OOF scores).
Rank Averaging (averaging the ranks rather than raw probabilities—this is more robust to calibration differences between models).
Power Mean (Geometric averaging).

How to Run:
Clone the repository:
git clone https://github.com/egor2025slo/loan_project_completed
Install dependencies:
pip install -r requirements.txt
Run the notebook:
Open competition_completed_well_done.ipynb in Jupyter Lab or Google Colab.
