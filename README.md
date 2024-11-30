# Income Classification using Machine Learning HOMEWWORK for ECON418: Intro to Econometrics @ University of Arizona.

This project involves predicting whether an individual's income exceeds $50,000 using various machine learning models. The dataset, preprocessing steps, model training, and evaluation are documented as part of an assignment for the **ECON 418-518: Introduction to Econometrics** course, Fall 2024.

## Project Overview
The project aims to:
1. Preprocess and clean the dataset for analysis.
2. Apply multiple machine learning models, including **Lasso Regression**, **Ridge Regression**, and **Random Forest**, to classify income levels.
3. Evaluate model performance using metrics like classification accuracy, confusion matrices, and other relevant statistics.
4. Address potential issues related to class imbalance.

---

## Dataset
- The dataset is sourced from the [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult).
- Features include demographic and employment-related variables such as age, education, marital status, hours per week, etc.
- The target variable is **income**, indicating whether an individual's income is `>50K` or `<=50K`.

---

## Steps Performed

### 1. Data Preprocessing
- Dropped irrelevant columns like `fnlwgt`, `occupation`, `relationship`, `capital-gain`, `capital-loss`, and `educational-num`.
- Converted categorical variables (e.g., `income`, `race`, `gender`) to binary indicators.
- Created additional features such as `age squared` and standardized numerical variables (`age`, `age squared`, `hours per week`).

### 2. Exploratory Data Analysis
- Calculated proportions for key variables:
  - Proportion of individuals earning > $50K: **23.93%**
  - Proportion of individuals working in the private sector: **69.42%**
  - Proportion of married individuals: **45.82%**
  - Proportion of females: **33.15%**

### 3. Model Training and Evaluation
#### (i) Lasso Regression
- Used 10-fold cross-validation to tune the shrinkage parameter (\( \lambda \)).
- Best \( \lambda \): **0.0518**
- Classification Accuracy: **81.32%**
- Variables with significant coefficients: `age`, `education`, `marital_status`, and `hours_per_week`.

#### (ii) Ridge Regression
- Used the same features as Lasso Regression for consistency.
- Classification Accuracy: **81.13%** (slightly lower than Lasso).

#### (iii) Random Forest
- Evaluated models with 100, 200, and 300 trees using 5-fold cross-validation.
- Best model: **Random Forest with 300 trees**.
- Training Accuracy: **82.68%**
- Testing Accuracy: **81.73%**

### 4. Confusion Matrix Analysis
- **Testing Data (300-tree Random Forest)**:
  - **True Positives** (correctly predicted ≤ $50K): **10,544**
  - **True Negatives** (correctly predicted > $50K): **1,432**
  - **False Positives** (predicted > $50K but actual ≤ $50K): **551**
  - **False Negatives** (predicted ≤ $50K but actual > $50K): **2,126**

---

## Key Takeaways
- The **Random Forest model with 300 trees** outperformed Lasso and Ridge regression models, achieving the highest accuracy on both the training and testing sets.
- The model struggled with class imbalance, evident from its high sensitivity (**95.03%**) but low specificity (**40.25%**). Techniques like rebalancing the dataset or adjusting the classification threshold could further improve performance.

---

## Project Files
- `data/`: Contains the dataset used for training and testing.
- `scripts/`: Includes R scripts for data preprocessing, model training, and evaluation.
- `README.md`: Project documentation (this file).

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MaximusAZ/ML-Income-Classifier-Using-UCI-Adult-Data-Set.git
   cd income-classification
