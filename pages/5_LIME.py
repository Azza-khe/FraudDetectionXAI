
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
from pathlib import Path
from scipy import sparse

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LIME - Explication locale",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Explication locale avec LIME")

st.write(
    """
    Cette page utilise LIME (Local Interpretable Model-agnostic Explanations)
    pour expliquer les prédictions du modèle XGBoost.

    LIME permet de comprendre quelles caractéristiques influencent
    une prédiction pour une transaction particulière.
    """
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"
DATA_PATH = ROOT / "data" / "processed" / "validation.parquet"


# =====================================================
# 2. CHARGEMENT DES FICHIERS
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


try:
    model = load_model()
    preprocessor = load_preprocessor()
    df = load_data()

except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.stop()


# =====================================================
# 3. PREPARATION DES DONNEES
# =====================================================

if "isFraud" not in df.columns:
    st.error("La colonne isFraud est absente du dataset.")
    st.stop()

y = df["isFraud"].astype(int)

excluded_columns = [
    "isFraud",
    "TransactionID",
    "TransactionDT",
    "CardGroup"
]

X = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

# Respecter exactement les colonnes utilisées
# lors de l'entraînement du préprocesseur.
if hasattr(preprocessor, "feature_names_in_"):
    X = X.reindex(
        columns=preprocessor.feature_names_in_
    )

if X.empty:
    st.error("Aucune caractéristique disponible.")
    st.stop()


# =====================================================
# 4. TRANSFORMATION DES DONNEES
# =====================================================

def to_dense(data):
    if sparse.issparse(data):
        return data.toarray()
    return np.asarray(data)


@st.cache_data
def prepare_lime_data(data, sample_size):
    sample = data.sample(
        n=min(sample_size, len(data)),
        random_state=42
    )

    transformed = preprocessor.transform(sample)

    return to_dense(transformed)


st.subheader("Préparation de l'explication")

sample_size = st.slider(
    "Nombre de transactions utilisées pour construire l'explication locale",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

try:
    X_lime = prepare_lime_data(X, sample_size)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        feature_names = [
            f"Feature_{i}"
            for i in range(X_lime.shape[1])
        ]

    if len(feature_names) != X_lime.shape[1]:
        feature_names = [
            f"Feature_{i}"
            for i in range(X_lime.shape[1])
        ]

except Exception as e:
    st.error(f"Erreur pendant la préparation des données : {e}")
    st.stop()


# =====================================================
# 5. CREATION DE L'EXPLIQUEUR LIME
# =====================================================

explainer = LimeTabularExplainer(
    training_data=X_lime,
    feature_names=feature_names,
    class_names=["Normal", "Fraude"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)


# =====================================================
# 6. SELECTION D'UNE TRANSACTION
# =====================================================

st.subheader("Sélection d'une transaction")

transaction_index = st.number_input(
    "Indice de la transaction dans le jeu de validation",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1
)

if st.button("Générer l'explication LIME", type="primary"):

    try:
        # Récupérer la transaction sélectionnée
        transaction = X.iloc[[int(transaction_index)]]

        # Transformer la transaction
        transaction_processed = to_dense(
            preprocessor.transform(transaction)
        )

        instance = transaction_processed[0]

        # Fonction de prédiction utilisée par LIME
        def predict_function(data):
            return model.predict_proba(data)

        # Probabilités du modèle
        probabilities = predict_function(
            transaction_processed
        )[0]

        fraud_probability = float(probabilities[1])
        normal_probability = float(probabilities[0])

        prediction = int(model.predict(transaction_processed)[0])
        actual = int(y.iloc[int(transaction_index)])

        # Générer l'explication locale
        explanation = explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_function,
            num_features=10,
            labels=(1,)
        )

        # =============================================
        # 7. AFFICHAGE DES RESULTATS
        # =============================================

        st.divider()
        st.subheader("Résultat de la prédiction")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Probabilité de fraude",
                f"{fraud_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Probabilité de transaction normale",
                f"{normal_probability * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Classe prédite",
                "Fraude" if prediction == 1 else "Normale"
            )

        st.write(
            "**Classe réelle :**",
            "Fraude" if actual == 1 else "Normale"
        )

        if prediction == actual:
            st.success("La prédiction est correcte.")
        else:
            st.warning("La prédiction est incorrecte.")

        # =============================================
        # 8. VISUALISATION LIME
        # =============================================

        st.divider()
        st.subheader("Explication locale LIME")

        explanation_list = explanation.as_list(label=1)

        explanation_df = pd.DataFrame(
            explanation_list,
            columns=["Caractéristique", "Contribution"]
        )

        explanation_df = explanation_df.sort_values(
            by="Contribution",
            ascending=True
        )

        fig = px.bar(
            explanation_df,
            x="Contribution",
            y="Caractéristique",
            orientation="h",
            title="Facteurs influençant la prédiction de fraude",
            labels={
                "Contribution": "Poids LIME",
                "Caractéristique": "Caractéristique"
            }
        )

        fig.add_vline(
            x=0,
            line_dash="dash"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.write("### Tableau des contributions")

        st.dataframe(
            explanation_df.sort_values(
                by="Contribution",
                ascending=False
            ),
            use_container_width=True
        )

        # =============================================
        # 9. INTERPRETATION
        # =============================================

        st.divider()
        st.subheader("Interprétation")

        st.write(
            """
            - Une contribution LIME positive favorise la classe Fraude (1).
            - Une contribution LIME négative favorise la classe Normale (0).
            - Plus la valeur absolue de la contribution est grande,
              plus son influence locale dans l'explication est importante.

            Attention : les poids LIME ne représentent pas des pourcentages
            de probabilité. Ils décrivent l'approximation locale du modèle.
            """
        )

        st.info(
            "LIME explique une transaction particulière. "
            "Il complète SHAP, qui permet également d'étudier "
            "l'importance des caractéristiques à l'échelle globale."
        )

    except Exception as e:
        st.error(f"Erreur lors de la génération de l'explication : {e}")
        st.exception(e)
