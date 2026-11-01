
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Détection des anomalies",
    page_icon="🚨",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

st.title("🚨 Détection des anomalies")
st.write(
    """
    Cette page utilise deux algorithmes non supervisés :
    **Isolation Forest** et **Local Outlier Factor (LOF)**.

    Leur objectif est de détecter les transactions présentant
    un comportement inhabituel, sans utiliser directement
    leur étiquette de fraude pendant la détection.
    """
)

# =====================================================
# 2. CHEMINS DES FICHIERS
# =====================================================

MODELS_PATH = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "validation.parquet"

# =====================================================
# 3. CHARGEMENT DES MODELES ET DONNEES
# =====================================================

@st.cache_resource
def load_models():

    features = joblib.load(
        MODELS_PATH / "anomaly_features.pkl"
    )

    imputer = joblib.load(
        MODELS_PATH / "anomaly_imputer.pkl"
    )

    scaler = joblib.load(
        MODELS_PATH / "anomaly_scaler.pkl"
    )

    isolation_forest = joblib.load(
        MODELS_PATH / "isolation_forest.pkl"
    )

    lof = joblib.load(
        MODELS_PATH / "lof_model.pkl"
    )

    return features, imputer, scaler, isolation_forest, lof


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


try:
    features, imputer, scaler, isolation_forest, lof = load_models()
    df = load_data()

except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()


# =====================================================
# 4. VERIFICATION DES DONNEES
# =====================================================

missing_features = [
    col for col in features
    if col not in df.columns
]

if missing_features:
    st.error(
        "Certaines caractéristiques sont absentes du dataset : "
        + ", ".join(missing_features)
    )
    st.stop()

if "isFraud" not in df.columns:
    st.error("La colonne isFraud est absente.")
    st.stop()


# =====================================================
# 5. PREPARATION DES DONNEES
# =====================================================

def prepare_data(data):

    X = data[features].copy()

    # Remplacer les valeurs infinies
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convertir les colonnes en valeurs numériques
    X = X.apply(pd.to_numeric, errors="coerce")

    # Imputation des valeurs manquantes
    X_imputed = imputer.transform(X)

    # Normalisation
    X_scaled = scaler.transform(X_imputed)

    return X_scaled


# =====================================================
# 6. PRESENTATION DES MODELES
# =====================================================

st.divider()
st.subheader("Les algorithmes utilisés")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌳 Isolation Forest")
    st.write(
        """
        Isolation Forest détecte les anomalies en isolant
        les observations qui sont différentes des autres.

        Une transaction facile à isoler est considérée
        comme potentiellement anormale.
        """
    )

with col2:
    st.markdown("### 📍 Local Outlier Factor (LOF)")
    st.write(
        """
        LOF compare la densité locale d'une transaction
        avec celle de ses voisins.

        Une transaction située dans une zone de faible densité
        par rapport à ses voisines peut être considérée
        comme anormale.
        """
    )


# =====================================================
# 7. SELECTION D'UNE TRANSACTION
# =====================================================

st.divider()
st.subheader("Sélection d'une transaction")

transaction_index = st.number_input(
    "Indice de la transaction dans le jeu de validation",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("Analyser la transaction", type="primary"):

    try:

        # Récupérer la transaction
        transaction = df.iloc[[int(transaction_index)]]

        # Préparer les données
        X_scaled = prepare_data(transaction)

        # =============================================
        # 8. PREDICTIONS DES DEUX MODELES
        # =============================================

        prediction_iso = int(
            isolation_forest.predict(X_scaled)[0]
        )

        prediction_lof = int(
            lof.predict(X_scaled)[0]
        )

        score_iso = float(
            isolation_forest.decision_function(X_scaled)[0]
        )

        score_lof = float(
            lof.decision_function(X_scaled)[0]
        )

        # =============================================
        # 9. AFFICHAGE DES RESULTATS
        # =============================================

        st.divider()
        st.subheader("Résultats de la détection")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### Isolation Forest")

            if prediction_iso == -1:
                st.error("⚠️ Anomalie détectée")
            else:
                st.success("✅ Transaction normale")

            st.metric(
                "Score de décision",
                f"{score_iso:.5f}"
            )

        with col2:

            st.markdown("### LOF")

            if prediction_lof == -1:
                st.error("⚠️ Anomalie détectée")
            else:
                st.success("✅ Transaction normale")

            st.metric(
                "Score de décision",
                f"{score_lof:.5f}"
            )

        # =============================================
        # 10. COMPARAISON
        # =============================================

        st.divider()
        st.subheader("Comparaison des deux algorithmes")

        if prediction_iso == -1 and prediction_lof == -1:

            st.warning(
                "Les deux modèles détectent une anomalie. "
                "La transaction mérite une vérification."
            )

        elif prediction_iso == 1 and prediction_lof == 1:

            st.success(
                "Les deux modèles considèrent la transaction comme normale."
            )

        else:

            st.info(
                "Les deux modèles ne sont pas d'accord. "
                "Une analyse complémentaire est nécessaire."
            )

        comparison_df = pd.DataFrame({
            "Algorithme": [
                "Isolation Forest",
                "LOF"
            ],
            "Résultat": [
                "Anomalie" if prediction_iso == -1 else "Normale",
                "Anomalie" if prediction_lof == -1 else "Normale"
            ],
            "Score de décision": [
                score_iso,
                score_lof
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            comparison_df,
            x="Algorithme",
            y="Score de décision",
            color="Résultat",
            title="Scores de décision des deux modèles",
            color_discrete_map={
                "Anomalie": "red",
                "Normale": "green"
            }
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Frontière de décision"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =============================================
        # 11. COMPARAISON AVEC LA CLASSE REELLE
        # =============================================

        st.divider()
        st.subheader("Vérification avec la classe réelle")

        actual = int(transaction["isFraud"].iloc[0])

        st.write(
            "**Classe réelle :**",
            "Fraude" if actual == 1 else "Normale"
        )

        st.caption(
            "Cette information sert uniquement à évaluer "
            "les résultats. Elle n'est pas utilisée par "
            "les algorithmes pour détecter les anomalies."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Isolation Forest**")

            if prediction_iso == -1 and actual == 1:
                st.success("Anomalie détectée sur une fraude.")

            elif prediction_iso == -1 and actual == 0:
                st.warning("Anomalie détectée sur une transaction normale.")

            elif prediction_iso == 1 and actual == 1:
                st.error("Fraude non détectée comme anomalie.")

            else:
                st.success("Transaction normale correctement identifiée.")

        with col2:

            st.write("**LOF**")

            if prediction_lof == -1 and actual == 1:
                st.success("Anomalie détectée sur une fraude.")

            elif prediction_lof == -1 and actual == 0:
                st.warning("Anomalie détectée sur une transaction normale.")

            elif prediction_lof == 1 and actual == 1:
                st.error("Fraude non détectée comme anomalie.")

            else:
                st.success("Transaction normale correctement identifiée.")

        # =============================================
        # 12. CARACTERISTIQUES DE LA TRANSACTION
        # =============================================

        st.divider()
        st.subheader("Caractéristiques de la transaction")

        features_df = transaction[features].T.reset_index()

        features_df.columns = [
            "Caractéristique",
            "Valeur"
        ]

        st.dataframe(
            features_df,
            use_container_width=True,
            hide_index=True
        )

        # =============================================
        # 13. INTERPRETATION
        # =============================================

        st.divider()
        st.subheader("Comment interpréter les résultats ?")

        st.write(
            """
            - **-1** : le modèle considère la transaction comme une anomalie.
            - **1** : le modèle considère la transaction comme normale.
            - Un score de décision négatif indique une anomalie
              selon la frontière de décision du modèle.
            - Un score positif indique une observation considérée
              comme normale.

            Les valeurs des scores d'Isolation Forest et de LOF
            ne sont pas directement comparables, car les deux
            algorithmes utilisent des méthodes différentes.
            """
        )

        st.info(
            "Une anomalie n'est pas nécessairement une fraude. "
            "Ces modèles détectent des comportements inhabituels "
            "qui peuvent ensuite être examinés par un analyste."
        )

    except Exception as e:

        st.error(
            f"Erreur pendant l'analyse de la transaction : {e}"
        )

        st.exception(e)
