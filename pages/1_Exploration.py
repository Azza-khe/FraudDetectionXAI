
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Exploration des données",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_data():
    """Charge le jeu de test préparé pendant le notebook."""

    data_path = ROOT / "data" / "processed" / "test.parquet"

    if not data_path.exists():
        return None

    return pd.read_parquet(data_path)


df = load_data()


# ============================================================
# 3. TITRE
# ============================================================

st.title("📊 Exploration des données")

st.write(
    """
    Cette page permet d'explorer les transactions bancaires
    du jeu de test utilisé pour évaluer les modèles.
    
    L'objectif est de comprendre la structure des données,
    la répartition des classes et les caractéristiques des transactions.
    """
)

st.divider()


# ============================================================
# 4. VÉRIFICATION DES DONNÉES
# ============================================================

if df is None:

    st.error(
        "Le fichier data/processed/test.parquet est introuvable. "
        "Vérifie que le jeu de test a bien été sauvegardé."
    )

    st.stop()


if "isFraud" not in df.columns:

    st.error(
        "La colonne isFraud est absente du fichier test.parquet."
    )

    st.stop()


# ============================================================
# 5. INFORMATIONS GÉNÉRALES
# ============================================================

st.header("1. Informations générales")

total_transactions = len(df)
total_features = len(df.columns)

fraud_count = int((df["isFraud"] == 1).sum())
normal_count = int((df["isFraud"] == 0).sum())

fraud_percentage = (
    fraud_count / total_transactions * 100
    if total_transactions > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Nombre de transactions",
    f"{total_transactions:,}".replace(",", " ")
)

col2.metric(
    "Nombre de variables",
    total_features
)

col3.metric(
    "Transactions normales",
    f"{normal_count:,}".replace(",", " ")
)

col4.metric(
    "Transactions frauduleuses",
    f"{fraud_count:,}".replace(",", " ")
)

st.info(
    f"Les fraudes représentent {fraud_percentage:.2f} % "
    "des transactions du jeu de test."
)

st.divider()


# ============================================================
# 6. APERÇU DES DONNÉES
# ============================================================

st.header("2. Aperçu du jeu de données")

st.write(
    "Voici les premières lignes du jeu de test après préparation."
)

st.dataframe(
    df.head(10),
    use_container_width=True
)

with st.expander("Afficher les types des variables"):
    types_df = pd.DataFrame({
        "Variable": df.columns,
        "Type": df.dtypes.astype(str).values
    })

    st.dataframe(types_df, use_container_width=True)

st.divider()


# ============================================================
# 7. RÉPARTITION DES CLASSES
# ============================================================

st.header("3. Répartition des transactions")

st.write(
    """
    La variable cible `isFraud` indique si une transaction est
    frauduleuse :

    - 0 : transaction normale.
    - 1 : transaction frauduleuse.
    """
)

class_df = pd.DataFrame({
    "Type de transaction": ["Normale", "Frauduleuse"],
    "Nombre": [normal_count, fraud_count]
})

col1, col2 = st.columns(2)

with col1:

    fig_pie = px.pie(
        class_df,
        names="Type de transaction",
        values="Nombre",
        title="Répartition des transactions",
        hole=0.4
    )

    st.plotly_chart(fig_pie, use_container_width=True)

with col2:

    fig_bar = px.bar(
        class_df,
        x="Type de transaction",
        y="Nombre",
        color="Type de transaction",
        title="Nombre de transactions par classe",
        text="Nombre"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

st.warning(
    "Le jeu de données est déséquilibré : les transactions normales "
    "sont beaucoup plus nombreuses que les transactions frauduleuses. "
    "L'accuracy seule ne suffit donc pas pour évaluer correctement "
    "un modèle de détection de fraude."
)

st.divider()


# ============================================================
# 8. ANALYSE DES MONTANTS
# ============================================================

st.header("4. Analyse des montants des transactions")

if "TransactionAmt" in df.columns:

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Montant moyen",
            f"{df['TransactionAmt'].mean():.2f}"
        )

        st.metric(
            "Montant médian",
            f"{df['TransactionAmt'].median():.2f}"
        )

    with col2:

        st.metric(
            "Montant minimum",
            f"{df['TransactionAmt'].min():.2f}"
        )

        st.metric(
            "Montant maximum",
            f"{df['TransactionAmt'].max():.2f}"
        )

    # Limitation à un échantillon pour accélérer l'affichage
    plot_df = df.sample(
        n=min(20000, len(df)),
        random_state=42
    ).copy()

    plot_df["Type de transaction"] = plot_df["isFraud"].map({
        0: "Normale",
        1: "Frauduleuse"
    })

    fig_amount = px.histogram(
        plot_df,
        x="TransactionAmt",
        color="Type de transaction",
        nbins=60,
        barmode="overlay",
        opacity=0.65,
        title="Distribution des montants des transactions"
    )

    st.plotly_chart(
        fig_amount,
        use_container_width=True
    )

else:

    st.warning("La variable TransactionAmt est absente.")


st.divider()


# ============================================================
# 9. VALEURS MANQUANTES
# ============================================================

st.header("5. Analyse des valeurs manquantes")

missing_count = df.isnull().sum()

missing_percentage = (
    missing_count / len(df) * 100
).sort_values(ascending=False)

missing_df = pd.DataFrame({
    "Variable": missing_percentage.index,
    "Valeurs manquantes (%)": missing_percentage.values,
    "Nombre de valeurs manquantes": missing_count[
        missing_percentage.index
    ].values
})

missing_df = missing_df[
    missing_df["Nombre de valeurs manquantes"] > 0
]

if len(missing_df) > 0:

    st.write(
        "Les variables suivantes contiennent encore des valeurs "
        "manquantes dans le jeu de test."
    )

    st.dataframe(
        missing_df.head(20),
        use_container_width=True
    )

else:

    st.success(
        "Aucune valeur manquante détectée dans ce jeu de test."
    )


st.divider()


# ============================================================
# 10. CONCLUSION
# ============================================================

st.header("6. Conclusion de l'exploration")

st.write(
    f"""
    Le jeu de test contient **{total_transactions:,} transactions**,
    dont **{fraud_count:,} transactions frauduleuses**.

    La proportion de fraudes est de **{fraud_percentage:.2f} %**.

    Cette répartition montre que le problème de détection de fraude
    est un problème de classification déséquilibrée. Il est donc
    important d'utiliser des métriques comme le rappel, la précision
    et la PR-AUC, en complément de la ROC-AUC.
    """.replace(",", " ")
)

st.success(
    "Exploration terminée. Vous pouvez maintenant consulter "
    "la page Modèles pour comparer les algorithmes."
)
