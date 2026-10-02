# Predicting Patient Survival Rates in Critically Ill Populations

**Author:** Nima Shadab   
**Contact:** nimashadab@hotmail.co.uk  

---

## 📋 Abstract
The objective of this project is to develop a machine learning model capable of making reliable predictions regarding the mortality outcome for critically ill patients within a specified time frame. Through systematic preprocessing, leak-free pipeline engineering, hyperparameter optimization, and adversarial validation, the final **CatBoost** model achieves a **7-fold Stratified Cross-Validation Macro F1-score of 0.761** and a **Public Leaderboard Macro F1-score of 0.754**.

---

## 📑 Table of Contents
- [Predicting Patient Survival Rates in Critically Ill Populations](#predicting-patient-survival-rates-in-critically-ill-populations)
  - [📋 Abstract](#-abstract)
  - [📑 Table of Contents](#-table-of-contents)
  - [🏥 Problem Overview](#-problem-overview)
    - [Missing Values Overview](#missing-values-overview)
  - [🛠️ Proposed Approach \& Architecture](#️-proposed-approach--architecture)
    - [Preprocessing Pipeline](#preprocessing-pipeline)
    - [Feature Engineering](#feature-engineering)
    - [Categorical Encoding](#categorical-encoding)
  - [⚙️ Model Selection \& Hyperparameter Tuning](#️-model-selection--hyperparameter-tuning)
    - [Selected Hyperparameters](#selected-hyperparameters)
  - [📊 Experimental Results](#-experimental-results)
    - [Model Performance](#model-performance)
    - [Adversarial Validation](#adversarial-validation)
  - [💡 Discussion \& Future Directions](#-discussion--future-directions)
  - [📚 References](#-references)

---

## 🏥 Problem Overview
The datasets contain clinical records collected between 1989–1991 and 1992–1994 across **5 medical centers in the USA**. The task is to predict whether a patient will survive or die in the future.

* **Dataset Split:**
  * **Training set:** 7,284 rows (patients)
  * **Test set:** 1,821 rows (patients)
  * **Feature Space:** 44 feature columns (excluding target `death` and `Id`)
* **Evaluation Metric:** **Macro F1 Score** is used as the evaluation metric to remain invariant to target class imbalance.
* **Key Observations:** Target distribution demonstrates class imbalance. Disease severity strongly correlates with mortality; for instance, approximately **89% of cancer patients** ended up dying, compared to **~60% for COPD/CHF/Cirrhosis patients**.

### Missing Values Overview
Several feature columns contain a significant number of missing values:

| Feature | Missing Count | Feature | Missing Count |
| :--- | :--- | :--- | :--- |
| `edu` | 1,298 | `wblc` | 168 |
| `income` | 2,389 | `pafi` | 1,868 |
| `charges` | 141 | `alb` | 2,698 |
| `totcst` | 710 | `bili` | 2,100 |
| `totmcst` | 2,782 | `crea` | 54 |
| `avtisst` | 69 | `ph` | 1,838 |
| `race` | 29 | `glucose` | 3,617 |
| `prg2m` | 1,306 | `bun` | 3,496 |
| `prg6m` | 1,292 | `urine` | 3,890 |
| `dnr` | 26 | `adlp` | 4,480 |
| `dnrday` | 26 | `adls` | 2,289 |

---

## 🛠️ Proposed Approach & Architecture

### Preprocessing Pipeline
To prevent data leakage, sklearn and custom transformers are chained into pipeline sequences tailored to model types:
* **Linear Models:** `FEAT NEW ENG` $\rightarrow$ `MYTRANSFORMER` $\rightarrow$ `TRANSFORMER W CAT` $\rightarrow$ `SIMPLEIMPUTER` $\rightarrow$ `STANDARDSCALER`.
* **Random Forest:** `FEAT NEW ENG` $\rightarrow$ `MYTRANSFORMER` $\rightarrow$ `TRANSFORMER W CAT`.
* **Boosted Trees:** `FEAT NEW ENG` $\rightarrow$ `MYTRANSFORMER` $\rightarrow$ `INCOME HANDLER`.

### Feature Engineering
The custom `FEAT NEW ENG` module applies the following transformations:
* **General Modifications:**
  * Rounded `edu` and `age` to mitigate overfitting on granular values.
  * Dropped `Id` (non-informative).
  * Dropped `temp_f` and `adls` due to perfect collinearity (\\(\rho = 1\\)) with `temp` and `adlsc`.
  * Replaced `admission_month` with cyclic features `month_sin` and `month_cos`.
* **Imputations:**
  * `alb`, `pafi`, `bili`, `crea`, `bun`, `wblc`, and `urine` imputed using domain baseline values.
  * `dnr` and `race` assigned explicit `"unknown"` category strings.
* **Domain-Specific Additions:**
  * **Modified Shock Index** (predicts hemorrhage severity in trauma).
  * **BUN/Creatinine Ratio** (`bun_crea`) for kidney health assessment.
  * **SIRS Score** (Systemic Inflammatory Response Syndrome).
  * **Renal Failure Index** (`rfi`).
  * **ADL Delta** (`adlsc` − `adlp`) to capture cognitive state and functional decline.
  * **Physician vs. Model Deltas:** Difference between physician estimates (`prg2m`, `prg6m`) and model estimates (`surv2m`, `surv6m`) to leverage direct physician insight.

### Categorical Encoding
* **`MYTRANSFORMER`:** Computes `sps` and `aps` score means per disease group (`dzgroup`) and calculates patient-level deltas to track disease progression while avoiding data leakage.
* **`TRANSFORMER W CAT`:**
  * **Ordinal Encoding:** Applied to `income`.
  * **One-Hot Encoding:** Applied to `sex`, `medical center`, `dzgroup`, `ca`, `dzclass`, `race`, and `dnr` for linear models and Random Forest.
  * **Ordinal `ca`:** Treated as ordinal (`no < yes < metastatic`) for XGBoost and LightGBM.

---

## ⚙️ Model Selection & Hyperparameter Tuning

Five distinct classifier families were evaluated: **CatBoost**, **XGBoost**, **LightGBM**, **Random Forest**, and **Logistic Regression**. Hyperparameter tuning was performed using **Optuna** paired with `TunedThresholdClassifierCV` across **7 stratified, shuffled folds**.

### Selected Hyperparameters

* **CatBoost:** `iterations = 4174`, `learning_rate = 0.004`, `depth = 6`, `l2_leaf_reg = 19`, `random_strength = 0.002`, `border_count = 214`, `bagging_temperature = 0.816`, `scale_pos_weight = 0.479`.
* **XGBoost:** `n_estimators = 4717`, `max_depth = 2`, `learning_rate = 0.005`, `max_bin = 195`, `min_child_weight = 1`, `gamma = 0.0002`, `reg_alpha = 0.0003`, `reg_lambda = 1.919`, `subsample = 0.910`, `colsample_bylevel = 0.853`, `colsample_bytree = 0.884`, `scale_pos_weight = 0.392`.
* **LightGBM:** `n_estimators = 3342`, `max_depth = 9`, `learning_rate = 0.003`, `max_bin = 238`, `num_leaves = 198`, `subsample = 0.882`, `subsample_freq = 2`, `reg_alpha = 0.809`, `reg_lambda = 0.042`, `scale_pos_weight = 0.493`, `min_data_in_leaf = 81`, `min_gain_to_split = 1.321`, `feature_fraction = 0.651`.
* **Random Forest:** `n_estimators = 4651`, `max_depth = 28`, `min_samples_leaf = 3`, `max_features = log2`, `class_weight = balanced`.
* **Logistic Regression:** `C = 0.052`, `l1_ratio = 0.852`, `solver = saga`, `max_iter = 5000`.

---

## 📊 Experimental Results

### Model Performance

| Model | Cross-Validation Score (Macro F1) | Public Leaderboard Score (Macro F1) |
| :--- | :---: | :---: |
| **CatBoost** | **0.761** | **0.754** |
| **XGBoost** | 0.758 | 0.752 |
| **LightGBM** | 0.758 | 0.752 |
| **Random Forest** | 0.754 | 0.747 |
| **Logistic Regression** | 0.751 | 0.741 |

The **CatBoost** classifier with an optimized decision threshold of **0.451** achieved the top performance across both cross-validation and the public leaderboard.

### Adversarial Validation
To ensure CV score stability without expensive nested cross-validation, **adversarial validation** was performed using a Random Forest classifier trained to distinguish between training and test sets (`IS_TEST` target):
* Features causing ROC-AUC scores to deviate from `0.5` by more than `0.1` were systematically evaluated and removed.
* This confirmed high feature distribution stability between training and test sets, validating the reliability of flat cross-validation scores.

---

## 💡 Discussion & Future Directions
* **Medical Knowledge Integration:** Further domain-based feature extraction and feature binning/categorization could enhance generalization.
* **Feature Selection:** Applying Recursive Feature Elimination (RFE) to prune noisy or redundant variables.
* **Ensembling:** Implementing model stacking across CatBoost, XGBoost, and LightGBM predictions.
* **Variance Tracking:** Tracking cross-validation standard deviation across folds to ensure low-variance model selection.

---

## 📚 References
1. F. Harrell, "SUPPORT2." UCI Machine Learning Repository, 1995. Source: Vanderbilt University Department of Biostatistics.
2. H. I. Kotb, A. Mamdouh, A. Abedalmohsen, and S. Abd El Mageed Mohammed, "Correlation between modified shock index and severity index in predicting outcome in patients with hemorrhagic shock," *Journal of Current Medical Research and Practice*, vol. 4, no. 3, pp. 231–236, 2019.
3. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD*, 2019.
4. J. Wainer and G. Cawley, "Nested cross-validation when selecting classifiers is overzealous for most practical applications," 2018.
5. Z. Zajac, "Adversarial validation." FastML.

