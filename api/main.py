
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "processed" / "test.parquet"

app = FastAPI(
    title="FraudDetectionXAI API",
    description=(
        "API de détection de fraude bancaire utilisant "
        "un modèle XGBoost et des techniques de Machine Learning."
    ),
    version="1.0.0"
)


# ============================================================
# 2. CHARGEMENT DES MODELES
# ============================================================

try:
    model = joblib.load(
        MODELS_DIR / "xgboost_model.pkl"
    )

    preprocessor = joblib.load(
        MODELS_DIR / "preprocessor.pkl"
    )

    config = joblib.load(
        MODELS_DIR / "business_config.pkl"
    )

    expected_features = list(
        preprocessor.feature_names_in_
    )

    DEFAULT_THRESHOLD = float(
        config.get("threshold", 0.45)
    )

except Exception as e:
    raise RuntimeError(
        f"Erreur lors du chargement des modèles : {e}"
    )


# ============================================================
# 3. STRUCTURE DES DONNEES RECUES
# ============================================================

class PredictionRequest(BaseModel):

    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Dictionnaire contenant les caractéristiques "
            "d'une transaction."
        )
    )

    threshold: float | None = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description="Seuil de décision facultatif."
    )


# ============================================================
# 4. FONCTION DE PREDICTION
# ============================================================

def predict_transaction(
    features: Dict[str, Any],
    threshold: float
):

    try:

        # Création d'un DataFrame contenant une transaction
        transaction = pd.DataFrame([features])

        # Suppression des colonnes inutiles
        columns_to_remove = [
            "isFraud",
            "TransactionID",
            "TransactionDT",
            "CardGroup"
        ]

        transaction = transaction.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        # Respecter les variables attendues par le préprocesseur
        transaction = transaction.reindex(
            columns=expected_features
        )

        # Prétraitement
        X_processed = preprocessor.transform(
            transaction
        )

        # Prédiction
        probabilities = model.predict_proba(
            X_processed
        )

        fraud_probability = float(
            probabilities[0, 1]
        )

        normal_probability = float(
            probabilities[0, 0]
        )

        # Application du seuil
        prediction = int(
            fraud_probability >= threshold
        )

        decision = (
            "FRAUDE POTENTIELLE"
            if prediction == 1
            else "NORMALE"
        )

        return {
            "fraud_probability": round(
                fraud_probability, 6
            ),
            "normal_probability": round(
                normal_probability, 6
            ),
            "fraud_probability_percent": round(
                fraud_probability * 100, 2
            ),
            "threshold": threshold,
            "prediction": prediction,
            "decision": decision
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Erreur pendant la prédiction : {str(e)}"
            )
        )


# ============================================================
# 5. ROUTE PRINCIPALE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Bienvenue dans l'API FraudDetectionXAI",
        "version": "1.0.0",
        "documentation": "/docs"
    }


# ============================================================
# 6. VERIFICATION DE L'API
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "OK",
        "message": "L'API fonctionne correctement.",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


# ============================================================
# 7. LISTE DES MODELES
# ============================================================

@app.get("/models")
def get_models():

    return {
        "models": [
            {
                "name": "XGBoost",
                "type": "Supervisé",
                "purpose": (
                    "Classification et détection de fraude"
                )
            },
            {
                "name": "Isolation Forest",
                "type": "Non supervisé",
                "purpose": "Détection des anomalies"
            },
            {
                "name": "Local Outlier Factor",
                "type": "Non supervisé",
                "purpose": "Détection des anomalies"
            }
        ],
        "main_model": "XGBoost"
    }


# ============================================================
# 8. VARIABLES ATTENDUES
# ============================================================

@app.get("/features")
def get_features():

    return {
        "number_of_features": len(expected_features),
        "features": expected_features
    }


# ============================================================
# 9. SEUIL DE DECISION
# ============================================================

@app.get("/threshold")
def get_threshold():

    return {
        "default_threshold": DEFAULT_THRESHOLD,
        "description": (
            "Une transaction est signalée si sa probabilité "
            "de fraude est supérieure ou égale au seuil."
        )
    }


# ============================================================
# 10. PREDICTION PERSONNALISEE
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    threshold = (
        request.threshold
        if request.threshold is not None
        else DEFAULT_THRESHOLD
    )

    result = predict_transaction(
        request.features,
        threshold
    )

    return result


# ============================================================
# 11. PREDICTION SUR LE JEU DE TEST
# ============================================================

@app.get("/predict-test/{index}")
def predict_test_transaction(
    index: int,
    threshold: float | None = Query(
        default=None,
        ge=0.01,
        le=0.99
    )
):

    if not DATA_PATH.exists():

        raise HTTPException(
            status_code=500,
            detail="Le fichier test.parquet est introuvable."
        )

    try:

        # Chargement du jeu de test
        df = pd.read_parquet(DATA_PATH)

        # Vérification de l'indice
        if index < 0 or index >= len(df):

            raise HTTPException(
                status_code=404,
                detail="Indice de transaction invalide."
            )

        # Sélection de la transaction
        transaction = df.iloc[[index]].copy()

        # Classe réelle : uniquement pour l'évaluation
        actual_class = None

        if "isFraud" in transaction.columns:

            actual_class = int(
                transaction["isFraud"].iloc[0]
            )

        # Suppression des colonnes inutiles
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

        # Respect des variables attendues
        X_transaction = X_transaction.reindex(
            columns=expected_features
        )

        # Prétraitement
        X_processed = preprocessor.transform(
            X_transaction
        )

        # Prédiction XGBoost
        probabilities = model.predict_proba(
            X_processed
        )

        fraud_probability = float(
            probabilities[0, 1]
        )

        normal_probability = float(
            probabilities[0, 0]
        )

        # Choix du seuil
        selected_threshold = (
            threshold
            if threshold is not None
            else DEFAULT_THRESHOLD
        )

        # Décision
        prediction = int(
            fraud_probability >= selected_threshold
        )

        decision = (
            "FRAUDE POTENTIELLE"
            if prediction == 1
            else "NORMALE"
        )

        # Résultat
        result = {
            "transaction_index": index,
            "fraud_probability": round(
                fraud_probability, 6
            ),
            "normal_probability": round(
                normal_probability, 6
            ),
            "fraud_probability_percent": round(
                fraud_probability * 100, 2
            ),
            "threshold": selected_threshold,
            "prediction": prediction,
            "decision": decision,
            "actual_class": actual_class
        }

        # Comparaison avec la classe réelle
        if actual_class is not None:

            result["correct_prediction"] = (
                prediction == actual_class
            )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur : {str(e)}"
        )
