
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SHAP Global",
    page_icon="🌍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "validation.parquet"
MODELS_DIR = ROOT / "models"


# ============================================================
# 2. TITRE
# ============================================================

st.title("🌍 Explicabilité globale avec SHAP")

st.write(
    "Cette page permet de comprendre le comportement général "
    "du modèle XGBoost et d'identifier les variables qui influencent "
    "le plus ses prédictions."
)

st.info(
    "Contrairement à SHAP local, qui explique une seule transaction, "
    "SHAP global analyse les contributions sur plusieurs transactions."
)


# ============================================================
# 3. CHARGEMENT DES DONNEES ET MODELES
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        MODELS_DIR / "xgboost_model.pkl"
    )

    preprocessor = joblib.load(
        MODELS_DIR / "preprocessor.pkl"
    )

    return model, preprocessor


@st.cache_data
def load_validation_data():

    return pd.read_parquet(DATA_PATH)


try:

    model, preprocessor = load_models()

    df = load_validation_data()

except Exception as e:

    st.error(
        f"Erreur lors du chargement des fichiers : {e}"
    )

    st.stop()


# ============================================================
# 4. PREPARATION DES DONNEES
# ============================================================

columns_to_remove = [
    "isFraud",
    "TransactionID",
    "TransactionDT",
    "CardGroup"
]

X = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

expected_features = list(
    preprocessor.feature_names_in_
)

X = X.reindex(
    columns=expected_features
)


# ============================================================
# 5. PARAMETRES D'ANALYSE
# ============================================================

st.divider()

st.subheader("1. Paramètres de l'analyse")

sample_size = st.slider(
    "Nombre de transactions à analyser",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

random_state = 42

st.write(
    f"Nombre de transactions sélectionnées : {sample_size}"
)


# ============================================================
# 6. CALCUL DES VALEURS SHAP
# ============================================================

@st.cache_resource
def load_shap_explainer():

    return joblib.load(
        MODELS_DIR / "shap_explainer.pkl"
    )


try:

    shap_explainer = load_shap_explainer()

    sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=random_state
    )

    X_processed = preprocessor.transform(sample)

    shap_values = shap_explainer.shap_values(
        X_processed
    )

    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values[0])

    else:

        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[:, :, 1]

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    if values.shape[1] != len(feature_names):

        feature_names = [
            f"Variable_{i}"
            for i in range(values.shape[1])
        ]

except Exception as e:

    st.error(
        f"Erreur pendant le calcul SHAP : {e}"
    )

    st.stop()


# ============================================================
# 7. IMPORTANCE GLOBALE DES VARIABLES
# ============================================================

st.divider()

st.subheader("2. Importance globale des variables")

importance = np.mean(
    np.abs(values),
    axis=0
)

importance_df = pd.DataFrame({
    "Variable": feature_names,
    "Importance SHAP": importance
})

importance_df = importance_df.sort_values(
    by="Importance SHAP",
    ascending=False
)

top_n = st.slider(
    "Nombre de variables à afficher",
    min_value=5,
    max_value=30,
    value=15
)

top_features = importance_df.head(top_n)

fig_importance = px.bar(
    top_features.sort_values("Importance SHAP"),
    x="Importance SHAP",
    y="Variable",
    orientation="h",
    title="Top des variables les plus importantes",
    text_auto=".4f"
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.write(
    "Une importance SHAP élevée indique qu'une variable "
    "contribue fortement aux prédictions du modèle, en moyenne."
)


# ============================================================
# 8. TABLEAU DES IMPORTANCES
# ============================================================

st.divider()

st.subheader("3. Classement des variables")

st.dataframe(
    importance_df.head(top_n).style.format({
        "Importance SHAP": "{:.5f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 9. INTERPRETATION
# ============================================================

st.divider()

st.subheader("4. Interprétation des résultats")

best_feature = importance_df.iloc[0]

st.success(
    f"La variable la plus importante dans cet échantillon est "
    f"**{best_feature['Variable']}**, avec une importance SHAP "
    f"moyenne de **{best_feature['Importance SHAP']:.5f}**."
)

st.markdown("""
### Comment interpréter cette analyse ?

- Une variable avec une importance élevée influence fortement
  les prédictions du modèle.
- Une importance faible signifie que la variable a, en moyenne,
  un impact plus limité.
- L'importance globale ne donne pas directement le sens de
  l'influence : pour cela, il faut examiner les contributions SHAP
  individuelles.

### Limite de l'analyse

Les résultats dépendent de l'échantillon sélectionné. Ici,
l'analyse est effectuée sur un échantillon aléatoire du jeu
de validation, avec une graine fixée à 42.
""")

st.caption(
    "Les valeurs SHAP présentées correspondent aux contributions "
    "sur le score brut du modèle XGBoost (log-odds)."
)
