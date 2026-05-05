"""
model_training.py
=================
Ansvarar för all modellträning i Diamond-projektet:
- Logistisk Regression
- KNN Classifier
- KNN Regressor
- PCA-pipeline (dimensionsreduktion + klassificering)

Alla funktioner returnerar den tränade modellen så att
den kan användas för utvärdering i notebooks & main.py.

Används av: alla notebooks & main.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ── Logistisk Regression ───────────────────────────────────────────────────────

def train_logistic_regression(
    X_train,
    y_train,
    max_iter: int = 1000,
    random_state: int = 42,
):
    """
    Tränar en Logistisk Regression-modell.

    Logistisk Regression är vår basmodell (baseline) —
    enkel, snabb och tolkbar. Passar binär klassificering
    som dyr/billig diamant.

    Parametrar:
        X_train:      skalade träningsfeatures
        y_train:      träningslabels (0 = billig, 1 = dyr)
        max_iter:     max antal iterationer (standard = 1000)
        random_state: reproducerbarhet (standard = 42)

    Returnerar:
        modell: tränad LogisticRegression
    """
    modell = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )
    modell.fit(X_train, y_train)
    print("✅ Logistisk Regression tränad")
    return modell


# ── KNN Classifier ─────────────────────────────────────────────────────────────

def train_knn_classifier(
    X_train,
    y_train,
    k: int = 5,
):
    """
    Tränar en KNN Classifier-modell.

    KNN klassificerar en diamant baserat på de k närmaste
    grannarna i träningsdatan. Känslig för skalning —
    därför används StandardScaler i data_processing.py.

    Parametrar:
        X_train: skalade träningsfeatures
        y_train: träningslabels (0 = billig, 1 = dyr)
        k:       antal grannar (standard = 5)

    Returnerar:
        modell: tränad KNeighborsClassifier
    """
    modell = KNeighborsClassifier(n_neighbors=k)
    modell.fit(X_train, y_train)
    print(f"✅ KNN Classifier tränad med k={k}")
    return modell


def find_best_k(
    X_train,
    X_test,
    y_train,
    y_test,
    k_range: range = range(1, 21),
):
    """
    Testar olika värden på k och returnerar accuracy för varje k.
    Används i notebook 03 för att hitta optimalt k.

    Parametrar:
        X_train:  skalade träningsfeatures
        X_test:   skalade testfeatures
        y_train:  träningslabels
        y_test:   testlabels
        k_range:  vilka k-värden som testas (standard = 1–20)

    Returnerar:
        k_värden:   lista med testade k
        accuracies: lista med accuracy per k
    """
    k_värden   = list(k_range)
    accuracies = []

    for k in k_värden:
        modell = KNeighborsClassifier(n_neighbors=k)
        modell.fit(X_train, y_train)
        acc = modell.score(X_test, y_test)
        accuracies.append(acc)

    bästa_k   = k_värden[np.argmax(accuracies)]
    bästa_acc = max(accuracies)
    print(f"✅ Bästa k = {bästa_k} med accuracy = {bästa_acc:.3f}")
    return k_värden, accuracies


# ── KNN Regressor ──────────────────────────────────────────────────────────────

def train_knn_regressor(
    X_train,
    y_train,
    k: int = 5,
):
    """
    Tränar en KNN Regressor-modell.

    Till skillnad från KNN Classifier förutsäger regressorn
    ett kontinuerligt värde — i det här fallet priset i USD.
    Utvärderas med RMSE och R² istället för Accuracy.

    Parametrar:
        X_train: skalade träningsfeatures
        y_train: faktiska priser (kontinuerliga värden)
        k:       antal grannar (standard = 5)

    Returnerar:
        modell: tränad KNeighborsRegressor
    """
    modell = KNeighborsRegressor(n_neighbors=k)
    modell.fit(X_train, y_train)
    print(f"✅ KNN Regressor tränad med k={k}")
    return modell


# ── PCA + Klassificering ───────────────────────────────────────────────────────

def train_pca_logistic(
    X_train,
    y_train,
    n_components: int = 2,
    max_iter: int = 1000,
    random_state: int = 42,
):
    """
    Tränar en pipeline: PCA → Logistisk Regression.

    PCA reducerar antalet dimensioner innan klassificering.
    Gör det möjligt att visualisera data i 2D och se om
    klasserna är separerbara.

    Parametrar:
        X_train:      skalade träningsfeatures
        y_train:      träningslabels
        n_components: antal PCA-komponenter (standard = 2)
        max_iter:     max iterationer för LogReg (standard = 1000)
        random_state: reproducerbarhet (standard = 42)

    Returnerar:
        pipeline: tränad Pipeline (PCA + LogisticRegression)
    """
    pipeline = Pipeline([
        ("pca",   PCA(n_components=n_components, random_state=random_state)),
        ("logreg", LogisticRegression(max_iter=max_iter, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    print(f"✅ PCA (n={n_components}) + Logistisk Regression tränad")
    return pipeline


def train_pca_knn(
    X_train,
    y_train,
    n_components: int = 2,
    k: int = 5,
):
    """
    Tränar en pipeline: PCA → KNN Classifier.

    Kombinerar dimensionsreduktion med KNN för att se
    om färre dimensioner förbättrar eller försämrar resultatet.

    Parametrar:
        X_train:      skalade träningsfeatures
        y_train:      träningslabels
        n_components: antal PCA-komponenter (standard = 2)
        k:            antal grannar för KNN (standard = 5)

    Returnerar:
        pipeline: tränad Pipeline (PCA + KNeighborsClassifier)
    """
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components)),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    pipeline.fit(X_train, y_train)
    print(f"✅ PCA (n={n_components}) + KNN (k={k}) tränad")
    return pipeline