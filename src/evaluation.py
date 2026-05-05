"""
evaluation.py
=============
Ansvarar för all modellutvärdering i Diamond-projektet:
- Skriver ut Classification Report
- Plottar Confusion Matrix
- Plottar ROC AUC-kurva
- Skriver ut regressionsmått (RMSE, R²)

Används av: alla notebooks & main.py
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
    r2_score,
)


# ── Klassificering ─────────────────────────────────────────────────────────────

def print_classification_report(y_test, y_pred, modellnamn: str = "Modell"):
    """
    Skriver ut Classification Report med Accuracy,
    Precision, Recall och F1-score.

    Parametrar:
        y_test:     faktiska värden
        y_pred:     modellens förutsägelser
        modellnamn: visningsnamn i utskriften
    """
    print(f"\n📊 Classification Report — {modellnamn}")
    print("─" * 50)
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Billig (0)", "Dyr (1)"]
    ))


def plot_confusion_matrix(y_test, y_pred, modellnamn: str = "Modell", spara: str = None):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Billig (0)", "Dyr (1)"],
                yticklabels=["Billig (0)", "Dyr (1)"], ax=ax)
    ax.set_title(f"Confusion Matrix — {modellnamn}")
    ax.set_xlabel("Förutsagd klass")
    ax.set_ylabel("Faktisk klass")
    plt.tight_layout()
    if spara:
        fig.savefig(spara, dpi=150, bbox_inches="tight")
        print(f"   Plot sparad till: {spara}")
    plt.show()


def plot_roc_curve(y_test, y_prob, modellnamn: str = "Modell", spara: str = None):
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Slumpmässig gissning")
    ax.set_title(f"ROC-kurva — {modellnamn}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if spara:
        fig.savefig(spara, dpi=150, bbox_inches="tight")
        print(f"   Plot sparad till: {spara}")
    plt.show()
    print(f"   AUC-värde: {auc:.3f}")
    return auc


# ── Regression ─────────────────────────────────────────────────────────────────

def print_regression_report(y_test, y_pred, modellnamn: str = "Modell"):
    """
    Skriver ut RMSE och R² för regressionsmodeller.

    Parametrar:
        y_test:     faktiska värden
        y_pred:     modellens förutsägelser
        modellnamn: visningsnamn i utskriften
    """
    rmse = root_mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 Regressionsresultat — {modellnamn}")
    print("─" * 50)
    print(f"   RMSE : {rmse:,.0f} USD")
    print(f"   R²   : {r2:.3f}")
    return rmse, r2


def plot_predicted_vs_actual(y_test, y_pred, modellnamn: str = "Modell"):
    """
    Plottar förutsagt pris mot faktiskt pris.
    En perfekt modell ger punkter längs diagonalen.

    Parametrar:
        y_test:     faktiska priser
        y_pred:     förutsagda priser
        modellnamn: visningsnamn i plottens titel
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", lw=1.5, label="Perfekt förutsägelse"
    )
    plt.title(f"Förutsagt vs Faktiskt pris — {modellnamn}")
    plt.xlabel("Faktiskt pris (USD)")
    plt.ylabel("Förutsagt pris (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Jämförelse ─────────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict):
    """
    Plottar ett stapeldiagram som jämför Accuracy och AUC
    för flera klassificeringsmodeller.

    Parameter:
        results: dictionary på formen:
            {
                "Logistisk Regression": {"accuracy": 0.85, "auc": 0.91},
                "KNN k=5":             {"accuracy": 0.83, "auc": 0.89},
            }
    """
    modeller   = list(results.keys())
    accuracies = [v["accuracy"] for v in results.values()]
    aucs       = [v["auc"] for v in results.values()]

    x = range(len(modeller))

    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], accuracies, width=0.4, label="Accuracy", color="steelblue")
    plt.bar([i + 0.2 for i in x], aucs,       width=0.4, label="AUC",      color="coral")
    plt.xticks(x, modeller, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.title("Modellöversikt — Accuracy & AUC")
    plt.ylabel("Poäng")
    plt.legend()
    plt.tight_layout()
    plt.show()