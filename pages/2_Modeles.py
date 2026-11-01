
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Comparaison des modèles",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs" / "metrics.json"


# ============================================================
# 2. TITRE
# ============================================================

st.title("📊 Comparaison des modèles")

st.write(
    "Cette page présente les performances des algorithmes "
    "utilisés pour la détection de fraude."
)

st.write("Trois modèles sont comparés :")

st.markdown("""
- **XGBoost** : apprentissage supervisé.
- **Isolation Forest** : détection non supervisée d'anomalies.
- **Local Outlier Factor (LOF)** : détection non supervisée des observations atypiques.
""")


# ============================================================
# 3. CHARGEMENT DES METRIQUES
# ============================================================

if not METRICS_PATH.exists():

    st.error(
        f"Le fichier des métriques est introuvable : {METRICS_PATH}"
    )

    st.stop()

try:

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

except Exception as e:

    st.error(f"Erreur de lecture du fichier JSON : {e}")
    st.stop()


# Récupération de la comparaison sur validation
validation_comparison = metrics.get(
    "validation_comparison",
    {}
)

if not validation_comparison:

    st.error(
        "Aucune comparaison sur validation n'est disponible."
    )

    st.write("Clés trouvées dans le fichier :", list(metrics.keys()))
    st.stop()


# Création du DataFrame
metrics_df = pd.DataFrame.from_dict(
    validation_comparison,
    orient="index"
)

metrics_df.index.name = "Modèle"
metrics_df = metrics_df.reset_index()

# Vérification des colonnes
required_columns = ["PR-AUC", "ROC-AUC"]

for column in required_columns:

    if column not in metrics_df.columns:

        st.error(
            f"La colonne {column} est absente du tableau."
        )

        st.write("Colonnes disponibles :", metrics_df.columns.tolist())
        st.stop()


# ============================================================
# 4. TABLEAU COMPARATIF
# ============================================================

st.divider()

st.subheader("1. Tableau comparatif")

st.dataframe(
    metrics_df.style.format({
        "PR-AUC": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 5. COMPARAISON PR-AUC
# ============================================================

st.divider()

st.subheader("2. Comparaison selon la PR-AUC")

st.write(
    "La PR-AUC mesure la qualité du compromis entre précision "
    "et rappel pour différentes valeurs du seuil de décision."
)

st.write(
    "Elle est particulièrement utile lorsque les classes sont "
    "déséquilibrées, comme dans la détection de fraude."
)

fig_pr = px.bar(
    metrics_df,
    x="Modèle",
    y="PR-AUC",
    color="Modèle",
    text_auto=".4f",
    title="Comparaison des modèles selon la PR-AUC",
    range_y=[0, 1]
)

fig_pr.update_layout(
    yaxis_title="PR-AUC",
    xaxis_title="Modèle",
    showlegend=False
)

st.plotly_chart(
    fig_pr,
    use_container_width=True
)


# ============================================================
# 6. COMPARAISON ROC-AUC
# ============================================================

st.divider()

st.subheader("3. Comparaison selon la ROC-AUC")

st.write(
    "La ROC-AUC mesure la capacité du modèle à distinguer "
    "les transactions normales des transactions frauduleuses."
)

st.write(
    "Une valeur proche de 1 indique une meilleure capacité "
    "de discrimination."
)

fig_roc = px.bar(
    metrics_df,
    x="Modèle",
    y="ROC-AUC",
    color="Modèle",
    text_auto=".4f",
    title="Comparaison des modèles selon la ROC-AUC",
    range_y=[0, 1]
)

fig_roc.update_layout(
    yaxis_title="ROC-AUC",
    xaxis_title="Modèle",
    showlegend=False
)

st.plotly_chart(
    fig_roc,
    use_container_width=True
)


# ============================================================
# 7. INTERPRETATION
# ============================================================

st.divider()

st.subheader("4. Interprétation des résultats")

st.markdown("""
### Pourquoi comparer ces trois algorithmes ?

- **XGBoost** apprend à partir des exemples étiquetés et utilise
  les informations disponibles pour prédire la classe d'une transaction.

- **Isolation Forest** recherche les observations qui peuvent être
  isolées rapidement dans l'espace des variables.

- **LOF** recherche les observations dont la densité locale est
  différente de celle de leurs voisins.
""")

st.warning(
    "Attention à l'interprétation : XGBoost a été entraîné sur un "
    "jeu beaucoup plus grand et avec 442 variables, tandis que les "
    "modèles non supervisés ont été entraînés sur des échantillons "
    "plus petits et 20 variables. La comparaison est donc indicative "
    "et ne constitue pas une comparaison parfaitement équilibrée."
)


# ============================================================
# 8. CONCLUSION
# ============================================================

st.divider()

st.subheader("5. Conclusion")

best_pr_model = metrics_df.loc[
    metrics_df["PR-AUC"].idxmax()
]

best_roc_model = metrics_df.loc[
    metrics_df["ROC-AUC"].idxmax()
]

st.write(
    f"Selon la PR-AUC, le meilleur modèle est "
    f"**{best_pr_model['Modèle']}**, avec une valeur de "
    f"**{best_pr_model['PR-AUC']:.4f}**."
)

st.write(
    f"Selon la ROC-AUC, le meilleur modèle est "
    f"**{best_roc_model['Modèle']}**, avec une valeur de "
    f"**{best_roc_model['ROC-AUC']:.4f}**."
)

st.success(
    "Dans cette expérimentation, XGBoost obtient les meilleures "
    "performances. Il sera utilisé comme modèle principal de "
    "prédiction, tandis qu'Isolation Forest et LOF serviront de "
    "méthodes complémentaires de détection d'anomalies."
)

st.info(
    "Comparaison terminée. Consultez maintenant la page "
    "Détection pour tester les prédictions."
)
