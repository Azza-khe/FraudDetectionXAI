
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import requests

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Détection de fraude",
    page_icon="🚨",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "test.parquet"
MODELS_DIR = ROOT / "models"

API_URL = "https://frauddetectionxai.onrender.com"


# ============================================================
# 2. TITRE
# ============================================================

st.title("🚨 Détection de fraude")

st.write(
    "Sélectionnez une transaction du jeu de test pour estimer "
    "son risque de fraude avec le modèle XGBoost."
)

st.info(
    "La prédiction est réalisée par l'API FastAPI. "
    "L'application Streamlit envoie la transaction à l'API "
    "et affiche le résultat retourné."
)


# ============================================================
# 3. CHARGEMENT DES ELEMENTS NECESSAIRES A SHAP
# ============================================================

@st.cache_resource
def load_explanation_resources():

    preprocessor = joblib.load(
        MODELS_DIR / "preprocessor.pkl"
    )

    shap_explainer = joblib.load(
        MODELS_DIR / "shap_explainer.pkl"
    )

    config = joblib.load(
        MODELS_DIR / "business_config.pkl"
    )

    return preprocessor, shap_explainer, config


# ============================================================
# 4. CHARGEMENT DU JEU DE TEST
# ============================================================

@st.cache_data
def load_test_data():

    return pd.read_parquet(DATA_PATH)


# ============================================================
# 5. VERIFICATION DE L'API
# ============================================================

def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:

        st.error(
            "Impossible de communiquer avec FastAPI. "
            "Vérifiez que le serveur API est démarré."
        )

        st.code(str(e))

        st.stop()


# ============================================================
# 6. CHARGEMENT DES FICHIERS
# ============================================================

try:

    preprocessor, shap_explainer, config = (
        load_explanation_resources()
    )

    df = load_test_data()

except Exception as e:

    st.error(
        f"Erreur lors du chargement des fichiers : {e}"
    )

    st.stop()


# ============================================================
# 7. VERIFICATION DE L'API
# ============================================================

api_status = check_api()

if api_status.get("status") == "OK":

    st.success(
        "API FastAPI connectée et opérationnelle."
    )

else:

    st.error(
        "L'API FastAPI ne fonctionne pas correctement."
    )

    st.stop()


# ============================================================
# 8. SEUIL DE DECISION
# ============================================================

default_threshold = float(
    config.get("threshold", 0.45)
)

threshold = st.slider(
    "Seuil de décision",
    min_value=0.01,
    max_value=0.99,
    value=default_threshold,
    step=0.01,
    help=(
        "Une transaction est signalée si sa probabilité "
        "de fraude est supérieure ou égale au seuil."
    )
)

st.caption(
    f"Seuil enregistré lors de l'optimisation : "
    f"{default_threshold:.2f}"
)


# ============================================================
# 9. SELECTION DE LA TRANSACTION
# ============================================================

st.divider()

st.subheader("1. Sélection d'une transaction")

transaction_index = st.number_input(
    "Numéro de la transaction dans le jeu de test",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

transaction_index = int(transaction_index)

transaction = df.iloc[[transaction_index]].copy()


# ============================================================
# 10. PREDICTION VIA FASTAPI
# ============================================================

st.divider()

st.subheader("2. Prédiction via l'API FastAPI")

if st.button(
    "Analyser la transaction",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Communication avec l'API et calcul de la prédiction..."
        ):

            response = requests.get(
                f"{API_URL}/predict-test/{transaction_index}",
                params={
                    "threshold": threshold
                },
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

        # Enregistrement du résultat dans la session
        st.session_state["api_prediction_result"] = result

        st.success(
            "Prédiction reçue avec succès depuis FastAPI."
        )

    except requests.exceptions.RequestException as e:

        st.error(
            "Erreur lors de la communication avec l'API."
        )

        st.code(str(e))

        st.stop()


# ============================================================
# 11. RECUPERATION DU RESULTAT
# ============================================================

result = st.session_state.get(
    "api_prediction_result"
)

# Vérifier que le résultat correspond à la transaction
# et au seuil actuellement sélectionnés

if (
    result is not None
    and (
        result["transaction_index"] != transaction_index
        or not np.isclose(result["threshold"], threshold)
    )
):

    result = None

    st.info(
        "Cliquez sur « Analyser la transaction » "
        "pour obtenir le nouveau résultat."
    )


if result is not None:

    fraud_probability = result["fraud_probability"]
    normal_probability = result["normal_probability"]
    prediction = result["prediction"]
    actual_class = result["actual_class"]


    # ========================================================
    # 12. AFFICHAGE DU RESULTAT
    # ========================================================

    st.divider()

    st.subheader("3. Résultat de la détection")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Probabilité de fraude",
            f"{result['fraud_probability_percent']:.2f} %"
        )

    with col2:

        st.metric(
            "Seuil de décision",
            f"{result['threshold'] * 100:.0f} %"
        )

    with col3:

        st.metric(
            "Transaction",
            str(transaction_index)
        )

    st.progress(
        fraud_probability,
        text="Niveau de risque estimé"
    )

    if prediction == 1:

        st.error(
            "🚨 Transaction suspecte : FRAUDE POTENTIELLE"
        )

        st.write(
            "La probabilité estimée est supérieure ou égale "
            "au seuil de décision."
        )

    else:

        st.success(
            "✅ Transaction classée comme NORMALE"
        )

        st.write(
            "La probabilité estimée est inférieure "
            "au seuil de décision."
        )


    # ========================================================
    # 13. INFORMATIONS SUR LA TRANSACTION
    # ========================================================

    st.divider()

    st.subheader("4. Informations sur la transaction")

    display_columns = [
        col for col in [
            "TransactionID",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "DeviceType",
            "DeviceInfo",
            "isFraud"
        ]
        if col in transaction.columns
    ]

    if display_columns:

        details = transaction[display_columns].T

        details.columns = ["Valeur"]

        st.dataframe(
            details,
            use_container_width=True
        )


    # ========================================================
    # 14. VERIFICATION AVEC LA CLASSE REELLE
    # ========================================================

    st.divider()

    st.subheader("5. Vérification avec la classe réelle")

    if actual_class is not None:

        col1, col2 = st.columns(2)

        with col1:

            if actual_class == 1:

                st.error("Classe réelle : FRAUDE")

            else:

                st.success("Classe réelle : NORMALE")

        with col2:

            if result.get("correct_prediction"):

                st.success("Prédiction correcte")

            else:

                st.warning("Prédiction incorrecte")

    else:

        st.warning(
            "La classe réelle n'est pas disponible."
        )


    # ========================================================
    # 15. EXPLICATION LOCALE AVEC SHAP
    # ========================================================

    st.divider()

    st.subheader(
        "6. Pourquoi le modèle a-t-il pris cette décision ?"
    )

    st.write(
        "SHAP permet d'identifier les variables qui ont "
        "contribué à la prédiction de cette transaction."
    )

    try:

        # Préparation des variables pour SHAP
        columns_to_remove = [
            "isFraud",
            "TransactionID",
            "TransactionDT",
            "CardGroup"
        ]

        X_transaction = transaction.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        expected_features = list(
            preprocessor.feature_names_in_
        )

        X_transaction = X_transaction.reindex(
            columns=expected_features
        )

        # Transformation des données
        X_processed = preprocessor.transform(
            X_transaction
        )

        # Calcul des valeurs SHAP
        shap_output = shap_explainer.shap_values(
            X_processed
        )

        # Gestion des différents formats SHAP
        if isinstance(shap_output, list):

            if len(shap_output) > 1:

                shap_values = np.asarray(
                    shap_output[1]
                )[0]

            else:

                shap_values = np.asarray(
                    shap_output[0]
                )[0]

        else:

            shap_output = np.asarray(shap_output)

            if shap_output.ndim == 3:

                shap_values = shap_output[0, :, 1]

            elif shap_output.ndim == 2:

                shap_values = shap_output[0]

            else:

                raise ValueError(
                    "Format des valeurs SHAP non reconnu."
                )

        # Noms des variables
        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        if len(feature_names) != len(shap_values):

            feature_names = [
                f"Variable_{i}"
                for i in range(len(shap_values))
            ]

        # Création du tableau des contributions
        shap_df = pd.DataFrame({

            "Variable": feature_names,

            "Contribution SHAP": shap_values

        })

        shap_df["Impact"] = np.where(
            shap_df["Contribution SHAP"] > 0,
            "Augmente le score de fraude",
            "Diminue le score de fraude"
        )

        shap_df["Importance"] = (
            shap_df["Contribution SHAP"].abs()
        )

        # Les 10 variables les plus influentes
        shap_top = shap_df.nlargest(
            10,
            "Importance"
        ).sort_values(
            "Contribution SHAP"
        )

        st.markdown(
            "### Les 10 variables les plus influentes"
        )

        fig_shap = px.bar(
            shap_top,
            x="Contribution SHAP",
            y="Variable",
            color="Impact",
            orientation="h",
            title=(
                "Explication SHAP de la transaction sélectionnée"
            ),
            color_discrete_map={
                "Augmente le score de fraude": "#E74C3C",
                "Diminue le score de fraude": "#2ECC71"
            }
        )

        fig_shap.update_layout(
            xaxis_title="Contribution SHAP (log-odds)",
            yaxis_title="Variable",
            legend_title="Influence"
        )

        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )

        st.markdown(
            "### Détail des contributions"
        )

        st.dataframe(
            shap_top[
                [
                    "Variable",
                    "Contribution SHAP",
                    "Impact"
                ]
            ].sort_values(
                "Contribution SHAP",
                key=lambda column: column.abs(),
                ascending=False
            ).style.format({
                "Contribution SHAP": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Une contribution SHAP positive augmente le score "
            "brut de fraude du modèle. Une contribution négative "
            "le diminue. Ces contributions ne représentent pas "
            "directement des variations en pourcentage "
            "de la probabilité."
        )

    except Exception as e:

        st.warning(
            f"Impossible de générer l'explication SHAP : {e}"
        )


    # ========================================================
    # 16. INTERPRETATION GENERALE
    # ========================================================

    st.divider()

    st.subheader("7. Interprétation")

    st.write(
        f"Le modèle XGBoost attribue à cette transaction "
        f"une probabilité de fraude de "
        f"{fraud_probability * 100:.2f} %."
    )

    if prediction == 1:

        st.write(
            "Cette transaction dépasse le seuil de décision. "
            "Elle est donc signalée comme potentiellement "
            "frauduleuse."
        )

    else:

        st.write(
            "Cette transaction ne dépasse pas le seuil "
            "de décision. Elle est donc classée comme normale "
            "par le modèle."
        )

    st.caption(
        "Attention : une prédiction est une estimation "
        "statistique. Une transaction signalée n'est pas "
        "nécessairement une fraude confirmée. "
        "Une vérification complémentaire peut être nécessaire."
    )

else:

    st.info(
        "Sélectionnez une transaction et cliquez sur "
        "« Analyser la transaction » pour commencer."
    )
