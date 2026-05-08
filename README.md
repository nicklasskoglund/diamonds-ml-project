# 💎 Diamond Prices — Supervised ML Project

> What determines the price of a diamond, and how well can machine learning predict it?

---

## About

We analyse a dataset of 53,772 diamonds and build models to:

- **Classify** diamonds as cheap or expensive (below/above the median price of $2,401)
- **Predict** the exact price in USD using a regression model

Key finding: **carat weight explains almost everything.** Simple models win on linearly
separable data — Logistic Regression with all features achieves 97.6% accuracy and
AUC 0.997, with no more advanced method beating it meaningfully.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nicklasskoglund/diamonds-ml-project.git
cd diamonds-ml-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This trains Logistic Regression, KNN Classifier and KNN Regressor,
evaluates them and displays a comparison.

### 5. Run notebooks (optional)

Launch Jupyter and open the notebooks in order (01 → 06):

```bash
jupyter notebook
```

---

## Project Structure

```
diamonds-ml-project/
│
├── data/
│   ├── diamonds.csv          # Raw data (53,772 diamonds)
│   └── diamonds_clean.csv    # Cleaned dataset (created by 01_EDA.ipynb)
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis & cleaning
│   ├── 02_logistic_regression.ipynb  # Logistic Regression (baseline)
│   ├── 03_knn_classifier.ipynb       # KNN Classifier + GridSearchCV
│   ├── 04_knn_regressor.ipynb        # KNN Regressor + residual analysis
│   ├── 05_pca_umap.ipynb             # Dimensionality reduction (PCA & UMAP)
│   └── 06_comparison_story.ipynb     # Summary & data story
│
├── outputs/
│   ├── eda/                    # Saved EDA plots (.png)
│   ├── models/                 # Saved plots (.png)
│   └── results/                # Model results as JSON (loaded by notebook 06)
│       ├── results_logreg.json
│       ├── results_knn_classifier.json
│       ├── results_knn_regressor.json
│       └── results_pca_umap.json
│
├── src/
│   ├── data_processing.py    # Cleaning, encoding, train/test split & scaling
│   ├── model_training.py     # Training functions for all models
│   └── evaluation.py         # Evaluation: report, confusion matrix, ROC curve
│
├── main.py                   # Runs the full pipeline in one go
├── requirements.txt          # Exact package versions
└── README.md
```

---

## Diamond Grading — The 4Cs

The dataset includes the four classic grading criteria that determine a diamond's
quality and price, in addition to its physical measurements.

### Cut
How well the diamond is cut — affects how it reflects light.

| Value | Description |
|---|---|
| `Fair` | Poorest — limited light reflection |
| `Good` | Acceptable cut |
| `Very Good` | Above average |
| `Premium` | Near ideal |
| `Ideal` | Best — maximum light reflection |

### Color
The degree of yellow tint in the diamond. Less color = higher value.

| Value | Description |
|---|---|
| `D` | Completely colorless — highest grade |
| `E` | Nearly colorless |
| `F` | Nearly colorless |
| `G` | Slight tint, difficult to detect |
| `H–J` | Noticeable yellow tint |

### Clarity
The amount of internal defects (inclusions) and surface flaws (blemishes).

| Value | Description |
|---|---|
| `IF` | Internally Flawless — no internal defects |
| `VVS1 / VVS2` | Very Very Slightly Included |
| `VS1 / VS2` | Very Slightly Included |
| `SI1 / SI2` | Slightly Included — visible under magnification |
| `I1` | Included — visible to the naked eye |

### Carat
The weight of the diamond. 1 carat = 0.2 grams. The strongest predictor of price
— carat correlates 0.92 with price in this dataset.

---

## Results

### Classification (cheap / expensive)

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression (9D) | **97.6%** | **0.997** |
| KNN optimised (9D) | 97.4% | 0.997 |
| KNN baseline k=5 (9D) | 96.9% | 0.993 |
| PCA n=5 + LogReg | 97.5% | 0.997 |
| UMAP n=2 + LogReg | 85.2% | 0.908 |

### Regression (exact price in USD)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| KNN Regressor optimised | $641 | $326 | 0.974 |
| KNN Regressor baseline | $703 | $373 | 0.968 |

---

## Key Insights

- **Carat explains almost everything** — correlation of 0.92 with price
- **Simpson's paradox** — better cut/color/clarity correlates *negatively* with price, because large heavy diamonds are often cut with lower quality but still sell for more
- **PCA with 5 components** retains 95.7% of variance — nearly identical results to 9D
- **UMAP degrades classification** on this dataset because the underlying structure is linear
- **Manhattan distance + distance weights** wins consistently for KNN, both for classification and regression

---

## 👥 Group Members

- Nicklas Skoglund
- Constantine Diamantis

---

## 🏫 Course Info

**Course:** Supervised Machine Learning  
**School:** Jensen Education