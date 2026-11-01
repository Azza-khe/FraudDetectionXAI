
from pathlib import Path
import json
import pickle

import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="FraudDetectionXAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. DESIGN PERSONNALISÉ
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(120deg, #10233f, #1e4770);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
    }

    .hero p {
        color: #dce8f5;
        font-size: 16px;
    }

    .section-title {
        color: #17365d;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e4eaf2;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. CHARGEMENT DES FICHIERS
# ============================================================

@st.cache_data
def load_metrics():
    path = ROOT / "outputs" / "metrics.json"

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_business_config():
    path = ROOT / "models" / "business_config.pkl"

    if not path.exists():
        return {}

    with open(path, "rb") as file:
        return pickle.load(file)


metrics = load_metrics()
business_config = load_business_config()

# Nouveau format JSON
test_metrics = metrics.get("test_metrics", {}).get("XGBoost", {})
validation_comparison = metrics.get("validation_comparison", {})

threshold = business_config.get("threshold", 0.45)
investigation_cost = business_config.get("investigation_cost", 5)


# ============================================================
# 4. BARRE LATÉRALE
# ============================================================

st.sidebar.markdown("## 🔍 FraudDetectionXAI")

st.sidebar.caption(
    "Plateforme intelligente de détection et d'explication "
    "des transactions frauduleuses."
)

st.sidebar.divider()

st.sidebar.markdown("### État du système")

st.sidebar.success("Fichiers de résultats chargés")

st.sidebar.metric("Seuil métier", f"{threshold:.2f}")

st.sidebar.divider()

st.sidebar.markdown("### Navigation")

st.sidebar.info(
    """
    Utilisez les pages du menu pour explorer les données,
    comparer les modèles, détecter les fraudes et consulter
    les explications SHAP.
    """
)

st.sidebar.divider()

st.sidebar.caption(
    "ENSI | Data Mining\n\n"
    "Challenge 2 — 2026/2027"
)


# ============================================================
# 5. EN-TÊTE
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🔍 FraudDetectionXAI</h1>
    <p>
        Application Web Interactive de Détection de Fraude Explicable
    </p>
    <p>
        Analyse intelligente • Machine Learning • Détection d'anomalies
        • Intelligence explicable
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. PRÉSENTATION
# ============================================================

st.markdown(
    '<h3 class="section-title">🎯 Présentation du projet</h3>',
    unsafe_allow_html=True
)

st.write(
    """
    FraudDetectionXAI est une application de détection de fraude
    basée sur le jeu de données IEEE-CIS Fraud Detection.

    Elle combine un modèle supervisé, des méthodes non supervisées
    et des techniques d'explicabilité afin d'identifier les
    transactions suspectes et de mieux comprendre les prédictions.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧠 Machine Learning")
    st.write(
        "Entraînement et comparaison de plusieurs algorithmes "
        "de détection."
    )

with col2:
    st.markdown("### 🚨 Détection")
    st.write(
        "Estimation du risque de fraude et génération d'alertes "
        "à partir d'un seuil."
    )

with col3:
    st.markdown("### 💡 Explicabilité")
    st.write(
        "Compréhension des facteurs qui influencent les décisions "
        "grâce à SHAP."
    )

st.divider()


# ============================================================
# 7. RÉSULTATS FINAUX SUR LE TEST
# ============================================================

st.markdown(
    '<h3 class="section-title">📊 Performance finale de XGBoost</h3>',
    unsafe_allow_html=True
)

st.caption(
    "Résultats obtenus sur le jeu de test indépendant, "
    "avec le seuil de décision retenu sur la validation."
)

if test_metrics:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "PR-AUC",
        f"{test_metrics.get('PR-AUC', 0):.4f}"
    )

    col2.metric(
        "ROC-AUC",
        f"{test_metrics.get('ROC-AUC', 0):.4f}"
    )

    col3.metric(
        "Précision",
        f"{test_metrics.get('Precision', 0):.2%}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Rappel",
        f"{test_metrics.get('Recall', 0):.2%}"
    )

    col5.metric(
        "F2-score",
        f"{test_metrics.get('F2-score', 0):.4f}"
    )

    col6.metric(
        "MCC",
        f"{test_metrics.get('MCC', 0):.4f}"
    )

else:
    st.warning(
        "Les métriques finales de test sont introuvables. "
        "Vérifiez outputs/metrics.json."
    )

st.divider()


# ============================================================
# 8. CONFIGURATION MÉTIER
# ============================================================

st.markdown(
    '<h3 class="section-title">⚙️ Paramètres de décision</h3>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Seuil de décision",
        f"{threshold:.2f}"
    )

with col2:
    st.metric(
        "Coût d'investigation estimé",
        f"{investigation_cost} unités"
    )

st.info(
    """
    Une transaction dont la probabilité prédite est supérieure
    ou égale au seuil est considérée comme une alerte potentielle
    de fraude.

    Le coût d'investigation et le seuil reposent sur des hypothèses
    métier définies dans le projet.
    """
)

st.divider()


# ============================================================
# 9. COMPARAISON SYNTHÉTIQUE
# ============================================================

st.markdown(
    '<h3 class="section-title">🤖 Comparaison des algorithmes</h3>',
    unsafe_allow_html=True
)

if validation_comparison:

    for model_name, values in validation_comparison.items():

        pr_auc = values.get("PR-AUC")
        roc_auc = values.get("ROC-AUC")

        if pr_auc is not None and roc_auc is not None:

            col1, col2, col3 = st.columns([2, 1, 1])

            col1.write(f"**{model_name}**")
            col2.metric("PR-AUC", f"{pr_auc:.4f}")
            col3.metric("ROC-AUC", f"{roc_auc:.4f}")

else:
    st.warning(
        "Les résultats de comparaison sur validation "
        "ne sont pas disponibles."
    )

st.caption(
    "Ces résultats proviennent de l'échantillon de validation "
    "de 20 000 transactions. Les conditions d'entraînement "
    "des modèles sont différentes."
)

st.divider()


# ============================================================
# 10. EXPLICATION DES MÉTRIQUES
# ============================================================

st.markdown(
    '<h3 class="section-title">📖 Comprendre les résultats</h3>',
    unsafe_allow_html=True
)

with st.expander("Que signifie la PR-AUC ?"):
    st.write(
        """
        La PR-AUC correspond à l'aire sous la courbe
        Précision-Rappel. Elle est particulièrement informative
        lorsque les fraudes sont rares.
        """
    )

with st.expander("Que signifie le rappel ?"):
    st.write(
        """
        Le rappel représente la proportion de fraudes réellement
        présentes que le modèle a réussi à détecter.
        """
    )

with st.expander("Que signifie la précision ?"):
    st.write(
        """
        La précision représente la proportion d'alertes émises
        qui correspondent réellement à des fraudes.
        """
    )

with st.expander("Pourquoi utiliser SHAP ?"):
    st.write(
        """
        SHAP permet d'expliquer les prédictions en estimant
        la contribution de chaque variable à la sortie du modèle.
        """
    )


# ============================================================
# 11. PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "FraudDetectionXAI — Projet académique de Data Mining | ENSI | 2026–2027"
)
