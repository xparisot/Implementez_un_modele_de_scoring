from flask import Flask, jsonify, request
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import os

# Ligne modification test yaml

# Chargement Flaskapp
app = Flask(__name__)
# Charger le modèle et les données
df_wk_path = 'df_wk.csv'
model_pipeline_path = 'model_pipeline_with_params.joblib'
df_wk = pd.read_csv(df_wk_path)
loaded_pipe_clf_params = joblib.load(model_pipeline_path)
loaded_model_pipeline = loaded_pipe_clf_params['pipeline']

# Liste des colonnes pour la prédiction
columns_for_prediction = ['Nombre_d_enfants', 'Montant_annuite', 'AGE',
   'Nombre_de_membres_dans_la_famille', 'Montant_total_du_credit',
   'Nombre_de_versements', 'montant_du_versement',
   'Montant_du_remboursement', 'Montant_du_crédit_precedent',
   'Type_de_contrat_Pret_d_especes', 'Femme', 'Homme',
   'Ne_possede_pas_de_vehicule', 'Possede_un_vehicule',
   'Possede_son_logement', 'Revenu_Non_salarie', 'Etudes_Superieures',
   'Marie', 'Habite_en_appartement', 'Pas_de_credit_en_cours',
   'Scoring_externe_1', 'Scoring_externe_2', 'Scoring_externe_3']

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Bienvenue sur l'API de prédiction!"})

@app.route('/full_client_data', methods=['GET'])
def get_full_client_data():
    client_id = request.args.get('client_id')
    
    try:
        client_id_int = int(client_id)
    except ValueError:
        return jsonify({"error": "Invalid Client ID"}), 400

    client_data = df_wk[df_wk['SK_ID_CURR'] == client_id_int]
    
    if client_data.empty:
        return jsonify({"error": "Client ID not found"}), 404
    
    # Convertir le DataFrame en dictionnaire
    client_data_dict = client_data.iloc[0].to_dict()
    
    return jsonify(client_data_dict)

@app.route('/average_values_all', methods=['GET'])
def get_average_values_all():
    # Calculer la moyenne pour chaque colonne
    average_values = df_wk.mean().to_dict()
    return jsonify(average_values)

@app.route('/prediction', methods=['GET'])
def get_prediction():
    client_id = request.args.get('client_id')
    client_data = df_wk[df_wk['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return jsonify({"error": "Client ID not found"}), 404
    
    # Transformez les données en un format que le modèle peut accepter
    input_data = client_data[columns_for_prediction]
    
    # Supprimer la colonne SK_ID_CURR 
    if 'SK_ID_CURR' in input_data.columns:
        input_data = input_data.drop(['SK_ID_CURR'], axis=1)
    
    # Faites la prédiction
    prediction = loaded_model_pipeline.predict(input_data)
    prediction_proba = loaded_model_pipeline.predict_proba(input_data)
    
    # Décision et probabilité arrondie
    decision = "accorde" if prediction[0] == 0 else "refuse"
    probability = round(prediction_proba[0][0] * 100, 2)
    
    return jsonify({"client_id": client_id, "decision": decision, "probability": probability})

@app.route('/info_client', methods=['GET'])
def get_info_client():
    client_id = request.args.get('client_id')
    client_data = df_wk[df_wk['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return jsonify({"error": "Client ID not found"}), 404
    
    age = client_data['AGE'].values[0]
    nombre_membres_famille = client_data['Nombre_de_membres_dans_la_famille'].values[0]
    nombre_d_enfants = client_data['Nombre_d_enfants'].values[0]
    possede_vehicule = client_data['Possede_un_vehicule'].values[0]
    possede_logement = client_data['Possede_son_logement'].values[0]

    return jsonify({"client_id": client_id, "age": age, "nombre_membres_famille": nombre_membres_famille,
                     "nombre_enfants": nombre_d_enfants, "Possede_un_vehicule": possede_vehicule, "Possede_son_logement": possede_logement})

@app.route('/info_banque', methods=['GET'])
def get_info_banque():   
    client_id = request.args.get('client_id')
    client_data = df_wk[df_wk['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return jsonify({"error": "Client ID not found"}), 404
    
    Score_externe_1 = client_data['Scoring_externe_1'].values[0]
    Score_externe_2 = client_data['Scoring_externe_2'].values[0]
    Score_externe_3 = client_data['Scoring_externe_3'].values[0]

    return jsonify({"Score_externe_1": Score_externe_1, "Score_externe_2": Score_externe_2, "Score_externe_3": Score_externe_3})

@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    feature_names = columns_for_prediction
    try:
        feature_importance = loaded_model_pipeline.named_steps['Classifier'].feature_importances_
        # Convertir les valeurs en float
        feature_importance_dict = {key: float(value) for key, value in zip(feature_names, feature_importance)}
    except AttributeError as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(feature_importance_dict)

@app.route('/average_values', methods=['GET'])
def get_average_values():
    attributes = ['AGE', 'Nombre_de_membres_dans_la_famille']
    average_values = {attr: df_wk[attr].mean() for attr in attributes}
    return jsonify(average_values)

# Créer un explainer LIME pour les données tabulaires (une seule fois)
explainer = LimeTabularExplainer(
    df_wk[columns_for_prediction].values,
    feature_names=columns_for_prediction,
    class_names=['Refusé', 'Accordé'],
    mode='classification'
)

@app.route('/client_feature_importance', methods=['GET'])
def get_lime_feature_importance():
    client_id = request.args.get('client_id')
    client_data = df_wk[df_wk['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return jsonify({"error": "Client ID not found"}), 404

    # Prédire avec le modèle
    input_data = client_data[columns_for_prediction].values[0]
    prediction = loaded_model_pipeline.predict_proba(input_data.reshape(1, -1))

    # Créer une nouvelle instance de l'explainer LIME pour chaque prédiction client
    exp = explainer.explain_instance(input_data, loaded_model_pipeline.predict_proba)

    # Obtenir l'importance des caractéristiques selon LIME pour ce client
    feature_importance = {exp.domain_mapper.feature_names[i]: weight for i, weight in exp.local_exp[1]}

    return jsonify(feature_importance)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
