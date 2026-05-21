# 🏠 House Prices – Regression & Feature Selection

A machine learning project built around the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition. The goal is to predict residential home sale prices using regression models, with a strong focus on preprocessing, feature engineering, and feature selection.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `0.0_Housing_Project.ipynb` | Task 1 – Introduction & data exploration |
| `0.1_Housing_Project.ipynb` | Task 2 – Data cleaning & encoding |
| `1_Housing_Project.ipynb` | Task 3 – Preprocessing pipelines |
| `2_Housing_Project.ipynb` | Task 4 – Baseline models |
| `3_Housing_Project.ipynb` | Task 5 – Full feature selection pipeline |
| `ULTIMATIVE_COMPETITION.ipynb` | Internal group competition submission |
| `ULTIMATIVE_kaggle_COMPETITION.ipynb` | Final Kaggle competition notebook |
| `house-prices-advanced-regression-techniques.zip` | Raw Kaggle dataset (train & test CSV) |
| `submission.csv` | Submission for the internal group competition |
| `sub_1.csv` | Submission for the Kaggle competition |

---

## 🔧 Workflow

1. **Data Loading & Exploration** – Load `train.csv`, inspect dtypes, missing values, and distributions
2. **Train-Test Split** – Split before any preprocessing to avoid data leakage
3. **Preprocessing** – Custom `ColumnTransformer` with:
   - Mean/mode imputation for numeric features
   - `OrdinalEncoder` for ordinal categorical features
   - `OneHotEncoder` for nominal categorical features
   - `StandardScaler` for numeric features
4. **Baseline Models** – `DecisionTreeRegressor` and `KNeighborsRegressor` wrapped in `sklearn` pipelines
5. **Feature Selection** – Four methods compared:
   - Variance Threshold
   - Collinearity Threshold (0.90)
   - SelectKBest (`f_regression`)
   - SelectFromModel
6. **Hyperparameter Tuning** – `GridSearchCV` over the best pipeline
7. **Kaggle Submission** – Predictions generated on `test.csv` and exported to `sub_1.csv`

---

## 🛠️ Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## 📊 Models Used

- `DecisionTreeRegressor`
- `KNeighborsRegressor`

Evaluated using **R² score** on a held-out test set and via **5-fold cross-validation** during grid search.

---

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
jupyter notebook
```

Open `ULTIMATIVE_kaggle_COMPETITION.ipynb` to see the full final pipeline.

---

## 📌 Notes

- The five numbered notebooks (`0.0` through `3`) represent the progressive build-up of the project across different tasks
- The `ULTIMATIVE_COMPETITION.ipynb` was used for an internal group competition before the Kaggle submission
- Dataset is from Kaggle and subject to their [competition rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)
