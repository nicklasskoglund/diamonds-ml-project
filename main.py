"""
main.py
=======
Kör hela Diamond-projektets pipeline i ett svep:

    1. Laddar och processar data      (data_processing.py)
    2. Tränar alla modeller           (model_training.py)
    3. Utvärderar och jämför modeller (evaluation.py)

Användning:
    python main.py

OBS: Kör notebooks för detaljerade experiment och visualiseringar.
Denna fil är till för att verifiera att hela pipelinen fungerar.
"""


from src.data_processing import run_pipeline
from src.model_training import (
    train_logistic_regression,
    train_knn_classifier,
    train_knn_regressor,
)
from src.evaluation import (
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    print_regression_report,
    plot_model_comparison,
)


# ── Konfiguration ──────────────────────────────────────────────────────────────

DATAFIL = "data/diamonds.csv"


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   💎 Diamond ML-projekt — Kör hela pipelinen")
    print("=" * 60)

    # ── 1. Data ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, features, scaler, df = run_pipeline(DATAFIL)

    # ── 2. Träna modeller ──────────────────────────────────────
    print("\n🔷 Tränar modeller...\n")

    logreg = train_logistic_regression(X_train, y_train)
    knn    = train_knn_classifier(X_train, y_train, k=5)

    # KNN Regressor tränas på faktiskt pris — inte price_label
    y_train_pris = df.loc[y_train.index, "price"]
    y_test_pris  = df.loc[y_test.index,  "price"]
    knn_reg = train_knn_regressor(X_train, y_train_pris, k=5)

    # ── 3. Utvärdera ───────────────────────────────────────────
    print("\n🔷 Utvärderar modeller...\n")

    # Logistisk Regression
    y_pred_logreg = logreg.predict(X_test)
    y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
    print_classification_report(y_test, y_pred_logreg, "Logistisk Regression")
    plot_confusion_matrix(y_test, y_pred_logreg, "Logistisk Regression")
    auc_logreg = plot_roc_curve(y_test, y_prob_logreg, "Logistisk Regression")

    # KNN Classifier
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    print_classification_report(y_test, y_pred_knn, "KNN k=5")
    plot_confusion_matrix(y_test, y_pred_knn, "KNN k=5")
    auc_knn = plot_roc_curve(y_test, y_prob_knn, "KNN k=5")

    # KNN Regressor
    y_pred_reg = knn_reg.predict(X_test)
    print_regression_report(y_test_pris, y_pred_reg, "KNN Regressor k=5")

    # ── 4. Jämförelse ──────────────────────────────────────────
    print("\n🔷 Jämför modeller...\n")

    results = {
        "Logistisk Regression": {
            "accuracy": logreg.score(X_test, y_test),
            "auc":      auc_logreg,
        },
        "KNN k=5": {
            "accuracy": knn.score(X_test, y_test),
            "auc":      auc_knn,
        },
    }

    plot_model_comparison(results)

    print("\n" + "=" * 60)
    print("   ✅ Pipeline klar!")
    print("=" * 60)


if __name__ == "__main__":
    main()