# Data Mining Assignment 2 — Spring 2026

## Student Information
- **Course:** Data Mining, NYCU Spring 2026
- **Assignment:** Assignment 2
- **Name:** Tymofii Voitekh (提姆西)
- **Student ID:** 111550203

---

## Repository Structure

```
├── model/
│   ├── linear_model.py        # Updated LinearModel (BaseEstimator compatible)
│   ├── metrics.py             # MSE, MAE, evaluate_linear_regression, evaluate_binary_classifier
│   ├── gradients.py           # MSE_grad, MAE_grad, logloss_sigmoid_grad
│   ├── activations.py         # Sigmoid activation
│   └── utils.py               # Helper functions
├── data/
│   ├── NYCU_Iris.csv          # Iris dataset (Assignment 1 / Task 1)
│   ├── mobile_price.csv       # Mobile price dataset (Tasks 2-5)
│   ├── linear_data/           # Linear regression datasets A-D
│   └── logistic_data/         # Logistic regression datasets A-D
├── Real_World_Classification.ipynb   # Task 1: K-fold Cross-Validation
├── Assignment2.ipynb                 # Tasks 2-5: SVM, ARM, PCA, K-Means
└── README.md
```

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend scipy
```

---

## Tasks Overview

### Task 1 — K-fold Cross-Validation (Real_World_Classification.ipynb)
- 5-fold cross-validation on preprocessed Iris data from Assignment 1
- Grid search over 16 hyperparameter combinations (lr × λ)
- Top 2 settings evaluated on test set

### Task 2 — SVM (Assignment2.ipynb)
- SVM classifier on mobile price dataset
- 60/20/20 train/val/test split with random_state=42
- C parameter sweep: {0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000}

### Task 3 — Association Rule Mining (Assignment2.ipynb)
- FP-Growth on price_range=1 subset
- Features: ram, int_memory, px_width, battery_power
- Frequent patterns (support ≥ 0.3) and association rules (support ≥ 0.3, confidence ≥ 0.4, lift ≥ 0.8)

### Task 4 — PCA and K-Means (Assignment2.ipynb)
- Z-score standardization → PCA visualization
- K-Means on all features vs. PCA 2D features
- Evaluated using Adjusted Rand Score

### Task 5 — ARM-Enhanced K-Means (Assignment2.ipynb)
- Per-class FP-Growth to compute feature discriminability scores
- Top-k discriminative features selected for K-Means
- Comparison with original K-Means across seeds = [0, 10, 42, 100, 999]
- Includes ablation study and Hungarian algorithm for cluster evaluation

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/tvoitekh/data-mining-assignment-2.git
cd data-mining-assignment-2
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend scipy
```

3. Open and run the notebooks in order:
   - Real_World_Classification.ipynb — Task 1
   - Assignment2.ipynb — Tasks 2 to 5

Note: Do not change any random_seed or random_state values as they affect evaluation results.

---

## Notes
- linear_model.py has been updated from Assignment 1 as the assignment desciption specified
- All random seeds are fixed as specified in the assignment instructions
- All results can be verified by running the notebooks top to bottom
