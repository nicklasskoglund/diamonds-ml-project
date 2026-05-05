"""
data_processing.py
==================
Ansvarar för all datahantering i Diamond-projektet:
- Laddar rådata från CSV
- Mappar om kategoriska graderingar till läsbara värden
- Kodar om ordinala variabler till numeriska
- Skalar numeriska features
- Delar upp data i tränings- och testset

Används av: alla notebooks & main.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Mappningar ────────────────────────────────────────────────────────────────

# Färg: J (sämst) → D (bäst), mappas till 1–7
COLOR_MAP = {
    "J": 1,  # Märkbart gul
    "I": 2,  # Lätt gulton
    "H": 3,  # Nästan färglös
    "G": 4,  # Nästan färglös, knappt märkbar
    "F": 5,  # Färglös
    "E": 6,  # Mycket färglös
    "D": 7,  # Perfekt färglös (bäst)
}

# Klarhet: I1 (sämst) → IF (bäst), mappas till 1–8
CLARITY_MAP = {
    "I1":  1,  # Synliga inneslutningar med blotta ögat
    "SI2": 2,  # Tydliga inneslutningar, syns med förstoringsglas
    "SI1": 3,  # Lätta inneslutningar, svåra att se
    "VS2": 4,  # Mycket lätta, kräver 10x förstoringsglas
    "VS1": 5,  # Mycket lätta, svåra även med glas
    "VVS2":6,  # Minimala inneslutningar, experter ser knappt
    "VVS1":7,  # Minimala inneslutningar, experter ser inte
    "IF":  8,  # Perfekt ren, inga inneslutningar
}

# Slipning: Fair (sämst) → Ideal (bäst), mappas till 1–5
CUT_MAP = {
    "Fair":      1,  # Grundläggande slipning
    "Good":      2,  # Bra slipning
    "Very Good": 3,  # Mycket bra slipning
    "Premium":   4,  # Premiumslipning
    "Ideal":     5,  # Perfekt slipning (bäst)
}


# ── Funktioner ────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Laddar CSV-filen och returnerar en DataFrame.

    Parameter:
        filepath: sökväg till diamonds.csv

    Returnerar:
        df: rådata som DataFrame
    """
    df = pd.read_csv(filepath)

    # Ta bort eventuell onödig indexkolumn från Kaggle
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"✅ Data laddad: {df.shape[0]} rader, {df.shape[1]} kolumner")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rensar data:
    - Tar bort dubbletter
    - Tar bort rader där x, y eller z är 0 (omöjliga mått)
    - Tar bort saknade värden

    Parameter:
        df: rådata

    Returnerar:
        df: rensat dataset
    """
    ursprunglig_storlek = len(df)

    # Ta bort dubbletter
    df = df.drop_duplicates()

    # Ta bort fysiskt omöjliga mätvärden (diamant kan inte ha mått = 0)
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    # Ta bort extrema y/z-värden som är troliga felregistreringar
    # En normal diamant är max ~20mm i någon dimension
    df = df[(df["y"] < 20) & (df["z"] < 20)]

    # Ta bort saknade värden
    df = df.dropna()

    borttagna = ursprunglig_storlek - len(df)
    print(f"✅ Rensning klar: {borttagna} rader borttagna, {len(df)} rader kvar")
    print(f"   (dubbletter, nollvärden & extrema mätvärden borttagna)")
    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappar om kategoriska variabler till läsbara siffror
    med hjälp av COLOR_MAP, CLARITY_MAP och CUT_MAP.

    Parameter:
        df: rensat dataset

    Returnerar:
        df: dataset med numeriska graderingar
    """
    df = df.copy()

    df["color"]   = df["color"].map(COLOR_MAP)
    df["clarity"] = df["clarity"].map(CLARITY_MAP)
    df["cut"]     = df["cut"].map(CUT_MAP)

    print("✅ Kategorier enkodade: cut, color, clarity → numeriska värden")
    return df


def create_price_label(df: pd.DataFrame, tröskelvärde: int = None) -> pd.DataFrame:
    """
    Skapar en binär kolumn 'price_label' för klassificering:
        1 = Dyr diamant (över tröskel)
        0 = Billig diamant (under eller lika med tröskel)

    Om inget tröskelvärde anges används medianen automatiskt.

    Parameter:
        df:           dataset
        tröskelvärde: prisgräns i USD (standard = median)

    Returnerar:
        df: dataset med ny kolumn 'price_label'
    """
    df = df.copy()

    if tröskelvärde is None:
        tröskelvärde = int(df["price"].median())

    df["price_label"] = (df["price"] > tröskelvärde).astype(int)

    antal_dyra   = df["price_label"].sum()
    antal_billiga = len(df) - antal_dyra

    print(f"✅ Prisgräns satt till ${tröskelvärde:,} (median)")
    print(f"   Dyra diamanter (1): {antal_dyra:,}")
    print(f"   Billiga diamanter (0): {antal_billiga:,}")
    return df


def split_and_scale(
    df: pd.DataFrame,
    target: str = "price_label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Delar upp data i tränings- och testset, sedan skalas features.

    OBS: StandardScaler fittas ENDAST på träningsdata för att
    undvika dataläckage till testsetet.

    Parametrar:
        df:           dataset med price_label
        target:       målvariabel (standard = 'price_label')
        test_size:    andel testdata (standard = 20%)
        random_state: reproducerbarhet (standard = 42)

    Returnerar:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
    """
    # Välj features — ta bort både pris och label
    features = [col for col in df.columns if col not in ["price", "price_label"]]

    X = df[features]
    y = df[target]

    # Dela upp i träning och test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Skala — fit på träning, transform på båda
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"✅ Datasplit: {len(X_train):,} träning / {len(X_test):,} test")
    print(f"   Features: {features}")
    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler


def run_pipeline(filepath: str, tröskelvärde: int = None) -> tuple:
    """
    Kör hela data-pipelinen i rätt ordning:
    load → clean → encode → label → split & scale

    Parameter:
        filepath:     sökväg till diamonds.csv
        tröskelvärde: valfri prisgräns (standard = median)

    Returnerar:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, df
    """
    print("\n🔷 Startar data-pipeline...\n")

    df = load_data(filepath)
    df = clean_data(df)
    df = encode_categories(df)
    df = create_price_label(df, tröskelvärde)

    X_train, X_test, y_train, y_test, features, scaler = split_and_scale(df)

    print("\n✅ Pipeline klar!\n")
    return X_train, X_test, y_train, y_test, features, scaler, df