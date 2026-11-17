# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave (churn) using Logistic Regression and Random Forest.

## Project Overview

Customer churn is expensive: keeping a customer costs much less than finding a new one. This project analyzes the Telco Customer Churn dataset to:

1. Explore how **monthly charges**, **tenure**, and **contract type** relate to churn
2. Build two classification models to predict churn
3. Compare the models using classification reports
4. Identify the features that matter most for churn

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** 7,043 customers, 21 columns
- **Target:** `Churn` (Yes/No)

## Tools Used

- Python
- pandas
- matplotlib and seaborn
- scikit-learn

## Project Steps

1. **Data cleaning**
   - Converted `TotalCharges` to numeric (11 blank values became missing and were dropped)
   - Removed `customerID`
   - Converted `Churn` to 1 (Yes) and 0 (No)
2. **Exploratory analysis**
   - Churn vs Monthly Charges
   - Churn vs Tenure
   - Churn vs Contract type
3. **Data preparation**
   - One-hot encoding of categorical columns (30 features)
   - 80/20 train-test split (stratified)
   - Feature scaling for Logistic Regression
4. **Modelling**
   - Logistic Regression
   - Random Forest
   - Both use `class_weight="balanced"` because churners are the minority class

## Key Findings

| Factor | Finding |
|---|---|
| Contract | [e.g. month-to-month customers churn most] |
| Tenure | [e.g. new customers churn more than long-term ones] |
| Monthly Charges | [e.g. churners tend to pay higher monthly charges] |

## Model Results

Evaluated on the test set (1,409 customers: 1,036 stayed, 373 churned).

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|---|---|---|---|
| Logistic Regression | 0.82 | 0.69 | 0.60 | 0.64 |
| Random Forest | 0.82 | 0.69 | 0.60 | 0.64 |

**Top features:** [list the top 3 from your charts]

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```
2. Install the libraries
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Download the dataset from Kaggle and place the CSV in the project folder
4. Run the script or open the notebook
   ```bash
   python churn_model.py
   ```

## Project Structure

```
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_model.ipynb
└── README.md
```

## Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Try other models such as XGBoost
- Adjust the decision threshold to catch more churners
- Add cross-validation

