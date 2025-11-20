# House Prices Prediction

## Project Overview
End-to-end Machine Learning project for predicting house prices. This project covers **data preprocessing**, **exploratory data analysis (EDA)**, **feature engineering**, **handling missing values**, **model training** (Random Forest & XGBoost), **evaluation with cross-validation**, and **feature importance analysis** to interpret key price drivers.

## Dataset
This project uses the **Ames Housing dataset** from Kaggle, which provides detailed information about residential homes in Ames, Iowa. It includes **79 explanatory variables** describing aspects such as size, quality, location, and year built.

Dataset link: [Ames Housing - Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

- `train.csv` — training data with target variable `SalePrice`
- `test.csv` — test data for predictions

## Quick Start

```bash
git clone https://github.com/AnaSlavina/house-prices-prediction.git
cd house-prices-prediction
pip install -r requirements.txt

# Place train.csv & test.csv into /data folder (from Kaggle)
jupyter notebook notebooks/          # Explore & reproduce

```

## Data Processing & EDA
1. **Target variable analysis:**  
  `SalePrice` has a right-skewed distribution, indicating extremely high-price outliers. To stabilize variance, reduce the influence of outliers, and improve model performance, a **log transformation** using `np.log1p()` was applied. This makes the distribution more symmetric and better suited for modeling.

2. **Feature correlation & importance:**  
  Identified the features most correlated with `SalePrice`. Columns with low impact or too many missing values were dropped.

3. **Handling missing values:**  
  - `LotFrontage` filled with median by neighborhood  
  - Garage & basement features handled logically (missing = no garage/basement)  
  - Other categorical features filled with the mode  
  - Numeric missing values filled with 0

4. **Feature Engineering:**  
  - `TotalSF` = Total living area (`TotalBsmtSF + 1stFlrSF + 2ndFlrSF`)  
  - `TotalPorchSF` = Total porch area (`OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`)
  - `HouseAge` = Age of the house (`YrSold - YearBuilt`)
  - `IsRemodeled` = Flag indicating if remodeled (`YearRemodAdd != YearBuilt`)
  - `TotalBath` = Total bathrooms (`FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`)  
  - `HasGarage` = Flag indicating if garage exists (`GarageArea > 0`)

5. **Encoding:**  
  One-hot encoding applied to categorical variables.

## Machine Learning

### Models
- **Random Forest Regressor**  
- **XGBoost Regressor**  

**Cross-Validation:**  
5-fold cross-validation was used to evaluate model performance reliably.

**Target Transformation:**  
`y = np.log1p(SalePrice)` for model training; predictions transformed back using `np.expm1()` for interpretability in real dollars.

### Model Evaluation
| Model           | CV RMSE (log) | CV RMSE (real $) |
|-----------------|----------------|-----------------|
| Random Forest   | 0.14775        | $30,367         |
| XGBoost         | 0.13661        | $26,225         |

### Feature Importance
Top features affecting house prices were identified using XGBoost's built-in feature importance.  
This highlights which variables the model relies on most for predictions, providing a basic level of interpretability.

## Results
- Log transformation improved model performance and stabilized variance  
- XGBoost achieved the best RMSE in both log scale and real dollar values  
- Submission file ready for Kaggle: `submissions/submission.csv`  

## Conclusion
The project successfully built predictive models (Random Forest and XGBoost) for house prices using the Ames Housing dataset.  
Cross-validation shows strong performance, with XGBoost achieving an RMSE of 0.1366 in log scale (~$26,225 in real prices).  
Feature importance analysis revealed the key drivers of house prices, including overall quality, living area, and garage features.  
The workflow demonstrates a complete end-to-end ML pipeline: data preprocessing, EDA, feature engineering, model training, evaluation, and interpretation.

The project successfully built predictive models (Random Forest and XGBoost) for house prices using the Ames Housing dataset.  
Cross-validation shows strong and stable performance, with XGBoost achieving **5-fold CV RMSE of 0.13183** (±0.018) in log scale — approximately **$25,495** average error in real prices.  
After submission, the model scored **0.1265** on the Kaggle Public Leaderboard (Top 20% out of 5,845 teams).  
Feature importance was visualized, highlighting the most influential predictors.  
The workflow demonstrates a complete, production-ready end-to-end ML pipeline: data loading → smart imputation → EDA → feature engineering → one-hot encoding → model training & tuning → cross-validation → submission → Streamlit deployment.


## Technologies & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

Contact: anastasiia.slavina.w@gmail.com

LinkedIn: [Anastasiia Slavina](https://www.linkedin.com/in/anastasiia-slavina/)
