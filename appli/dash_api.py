import pandas as pd
import os
import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import dash_table
from dash.dash_table.Format import Group
import plotly.graph_objs as go
import plotly.express as px
import requests
import dash_bootstrap_components as dbc
from dash import html

# Base URL for the Heroku API
BASE_URL = "https://api2-398a526923d8.herokuapp.com"
#BASE_URL = "http://127.0.0.1:5010"

# Définition de la fonction pour obtenir les information client
def get_client_info_details(client_id):
    response = requests.get(f"{BASE_URL}/full_client_data?client_id={client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Définition de la fonction pour obtenir la prédiction
def get_client_prediction(client_id):
    response = requests.get(f'{BASE_URL}/prediction?client_id={client_id}')
    if response.status_code == 200:
        prediction_data = response.json()
        decision = prediction_data['decision']
        probability = prediction_data['probability']
        return decision, probability
    else:
        return None, None

# Fonction pour obtenir les informations du tableau client
def get_client_info(client_id):
    response = requests.get(f'{BASE_URL}/info_client?client_id={client_id}')
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
    
# Fonction pour obtenir les informations banque
def get_info_banque(client_id):
    response = requests.get(f'{BASE_URL}/info_banque?client_id={client_id}')
    if response.status_code == 200:
        return response.json()
    else:
        return None   
    
# Fonction pour appeler les valeurs de la table du client 
def get_full_client_data(client_id):
    response = requests.get(f'{BASE_URL}/full_client_data?client_id={client_id}')
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def fetch_average_values_all():
    response = requests.get(f'{BASE_URL}/average_values_all')
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_client_feature_importance(client_id):
    response = requests.get(f'{BASE_URL}/client_feature_importance?client_id={client_id}')
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
    
# Initialisation de l'application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Etude Accord Crédit"

# Mise en page de l'application
app.layout = dbc.Container([

    # En-tête avec le titre et le logo
    dbc.Row([
        dbc.Col(html.Img(src="/static/logo.jpg", height="30px"), width=2),
        dbc.Col(html.H1("Etude Accord Crédit", style={'text-align': 'center'}), width=10)
    ], className="mb-4"),

   # Recherche par ID client
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Input(id='input-id', type='number', placeholder="Entrez l'ID client...", n_submit=0),
                dbc.Button('Rechercher', id='submit-button', color="primary", className="mt-2")
            ])
        ], width=12)
    ], className="mb-4"),

# Informations du client et Tableau d'informations client
dbc.Row([
    # Informations du client
    dbc.Col(dbc.Card([
        dbc.CardHeader("Informations du client"),
        dbc.CardBody(html.Div(id='client-info-details'))
    ]), width=6),

    # Tableau d'informations du client
    dbc.Col(dbc.Card([
        dbc.CardHeader("Tableau d'informations client"),
        dbc.CardBody(html.Div(id='client-info-table'))
    ]), width=6)
], className="mb-4"),

    # Informations et décisions de la banque
dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader("Décisions de la banque"),
        dbc.CardBody(html.Div(id='banque-info'))
    ]), width=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader("Données du client"),
        dbc.CardBody(dcc.Graph(id='client-bar-plot'))
    ]), width=6)
], className="mb-4"),

    # Onglets pour les graphiques
    dbc.Tabs([
        dbc.Tab(dcc.Graph(id='client-vs-average-chart'), label="Comparaison avec la moyenne des clients"),
        dbc.Tab(dcc.Graph(id='feature-importance-chart'), label="Importance des valeurs dans la décision"),
        dbc.Tab(dcc.Graph(id='negative-feature-importance-chart'), label="Valeurs avec un impact négatif"),
        dbc.Tab(dcc.Graph(id='positive-feature-importance-chart'), label="Valeurs avec un impact positif"),
    ], className="mb-4"),

    # Tableau d'importance des features et caractéristiques à impact positif
    dbc.Row([
    # Feature Importance Table
        dbc.Col(dbc.Card([
            dbc.CardHeader("Feature Importance Table"),
            dbc.CardBody(html.Div(id='feature-importance-table'))
        ]), width=6),
    # Caractéristiques à impact positif
        dbc.Col(dbc.Card([
            dbc.CardHeader("Caractéristiques à impact négatif"),
            dbc.CardBody(html.Div(id='negative-impact-features'))
        ]), width=6)
    ], className="mb-4"),

], fluid=True)


@app.callback(
    Output('client-info-table', 'children'),  
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')  
)
def update_client_info_table(n_submit, n_clicks, client_id):
    if client_id is None:
        return "Veuillez entrer un ID client valide."

    client_data = get_full_client_data(client_id)
    if client_data is None:
        return "ID client non trouvé."

    attributes = [key for key in client_data.keys() if key != "SK_ID_CURR"]
    client_values = [client_data[attr] for attr in attributes]

    table = dash_table.DataTable(
        columns=[{"name": "Caractéristique", "id": "feature"},
                 {"name": "Valeur", "id": "value"}],
        data=[{"feature": attr, "value": val} for attr, val in zip(attributes, client_values)],    
        style_table={'height': '300px', 'overflowY': 'auto', 'width': '50%'},  
        style_cell={'textAlign': 'left'},  
        page_size=10  
    )

    return table

@app.callback(
    Output('client-info-details', 'children'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value') 
)
def update_client_info_details(n_submit, n_clicks, client_id):
    if not client_id:
        return "Veuillez entrer un ID client valide."

    response = requests.get(f'{BASE_URL}/full_client_data?client_id={client_id}')
    data = response.json()

    if 'error' in data:
        return data['error']

    return [
        html.Div(f"Age du client: {data['AGE']}"),
        html.Div(f"Taille de la famille: {data['Nombre_de_membres_dans_la_famille']}"),
        html.Div(f"Nombre d'enfant: {data['Nombre_d_enfants']}"),
        html.Div(f"Marié: {'Oui' if data['Marie'] == 1 else 'Non'}"),
        html.Div(f"Possède une voiture: {'Oui' if data['Possede_un_vehicule'] == 1 else 'Non'}"),
        html.Div(f"Possède un logement: {'Oui' if data['Possede_son_logement'] == 1 else 'Non'}"),
        html.Div(f"Etudes supérieures: {'Oui' if data['Etudes_Superieures'] == 1 else 'Non'}"),
        html.Div(f"Crédit en cours: {'Oui' if data['Pas_de_credit_en_cours'] == 1 else 'Non'}")
    ]

@app.callback(
    Output('banque-info', 'children'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_info_banque(n_submit, n_clicks, client_id):
    if client_id is None:
        return "Veuillez entrer un ID client valide."
    
    # Obtenir les informations du client et la prédiction
    info_banque = get_info_banque(client_id)
    decision, probability = get_client_prediction(client_id)

    # Déterminer la décision et les couleurs en fonction de la probabilité
    if probability <= 45:
        decision_text = 'Refusé'
        colors = ['red', 'green']
        labels = [f"Refusé: {probability:.2f}%", f"Accordé: {100 - probability:.2f}%"]
    elif 45 < probability <= 55:
        decision_text = 'Intervention du conseiller'
        colors = ['orange', 'green']
        labels = [f"Intervention: {probability:.2f}%", f"Accordé: {100 - probability:.2f}%"]
    else:
        decision_text = 'Accordé'
        colors = ['red', 'green']
        labels = [f"Refusé: {100 - probability:.2f}%", f"Accordé: {probability:.2f}%"]
    
    values = [100 - probability, probability]
    
    fig = go.Figure(go.Pie(values=values, labels=labels, marker_colors=colors))
    fig.update_traces(textinfo='label+percent')

    camembert = dcc.Graph(figure=fig)

    response = [
        html.P(f"Score externe 1 : {info_banque['Score_externe_1']}"),
        html.P(f"Score externe 2 : {info_banque['Score_externe_2']}"),
        html.P(f"Score externe 3 : {info_banque['Score_externe_3']}"),
        html.P(f"Décision: {decision_text}"),
        html.P(f"Probabilité: {probability}%"),
        camembert
    ]

    return response

@app.callback(
    Output('client-bar-plot', 'figure'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_client_bar_plot(n_submit, n_clicks, client_id):
    if client_id is None:
        return go.Figure()

    # Obtenir les données complètes du client
    client_data = get_full_client_data(client_id)

    # Ignorer les colonnes non numériques ou celles qui n'ont pas de sens pour le graphique
    ignored_columns = ['SK_ID_CURR']  # <- Liste explicitement les colonnes à ignorer
    for col in ignored_columns:
        client_data.pop(col, None)

    # Extraire les clés et les valeurs
    keys = list(client_data.keys())
    values = list(client_data.values())

    # Créer le graphique à barres avec un axe y logarithmique
    fig = go.Figure(
        data=[go.Bar(x=keys, y=values)],
        layout=dict(
            title="Données du client en échelle logarithmique",
            yaxis=dict(type='log', title='Valeur')
        )
    )

    return fig

@app.callback(
    Output('feature-importance-chart', 'figure'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_feature_importance_chart(n_submit, n_clicks, client_id):
    if client_id is None:
        return go.Figure()  # Retourner un graphique vide si l'ID client n'est pas valide

    # Obtenir l'importance des caractéristiques de l'API
    feature_importance = get_client_feature_importance(client_id)

    if feature_importance is None:
        return go.Figure()  # Retourner un graphique vide si l'importance des caractéristiques n'est pas trouvée

    # Calculez la somme totale des valeurs absolues de l'importance
    total_importance = sum(abs(value) for value in feature_importance.values())
    
    # Convertir les valeurs d'importance en pourcentages
    feature_importance_percentage = {key: (value / total_importance) * 100 for key, value in feature_importance.items()}

    # Créer un DataFrame pour le graphique
    feature_importance_df = pd.DataFrame(list(feature_importance_percentage.items()), columns=['Feature', 'Percentage Importance'])
    feature_importance_df = feature_importance_df.sort_values(by="Percentage Importance", ascending=True)

    # Créer le graphique à barres avec Plotly Express
    fig = px.bar(feature_importance_df, x='Percentage Importance', y='Feature', title='Pourcentage d\'importance des caractéristiques', orientation='h')

    return fig

@app.callback(
    Output('negative-feature-importance-chart', 'figure'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_negative_feature_importance_chart(n_submit, n_clicks, client_id):
    if client_id is None:
        return go.Figure() 

    feature_importance = get_client_feature_importance(client_id) # Modifié ici

    if feature_importance is None:
        return go.Figure()  # Retourner un graphique vide si l'importance des caractéristiques n'est pas trouvée

    negative_features = {key: value for key, value in feature_importance.items() if value > 0}
    feature_importance_df = pd.DataFrame(list(negative_features.items()), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=True)

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', title='Valeurs avec un impact négatif', orientation='h')
    return fig

@app.callback(
    Output('positive-feature-importance-chart', 'figure'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_positive_feature_importance_chart(n_submit, n_clicks, client_id):
    if client_id is None:
        return go.Figure() 

    feature_importance = get_client_feature_importance(client_id)

    if feature_importance is None:
        return go.Figure()  # Return an empty plot if the feature importance isn't found

    positive_features = {key: value for key, value in feature_importance.items() if value < 0}
    feature_importance_df = pd.DataFrame(list(positive_features.items()), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', title='Valeurs avec un impact positif', orientation='h')
    return fig

@app.callback(
    Output('client-vs-average-chart', 'figure'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_comparison_chart(n_submit, n_clicks, client_id):
    if client_id is None:
        return go.Figure()  # Retourner un graphique vide si l'ID client n'est pas valide

    # Obtenir les informations du client
    client_info = get_full_client_data(client_id) # Modifiez cette fonction si nécessaire pour qu'elle renvoie un dictionnaire

    # Obtenir les valeurs moyennes
    average_values = fetch_average_values_all() # Assurez-vous que cette fonction renvoie un dictionnaire

    # Supprimer SK_ID_CURR des dictionnaires
    client_info.pop('SK_ID_CURR', None)
    average_values.pop('SK_ID_CURR', None)

    # Créer le graphique à barres avec Plotly
    fig = go.Figure(data=[
        go.Bar(name='Client', x=list(client_info.keys()), y=list(client_info.values())),
        go.Bar(name='Moyenne', x=list(average_values.keys()), y=list(average_values.values()))
    ])

    # Modifier la mise en page du graphique si nécessaire
    fig.update_layout(
        barmode='group',
        title='Comparaison des valeurs du client avec la moyenne des clients',
        yaxis=dict(type='log', title='Valeur (échelle logarithmique)')
    )

    return fig

@app.callback(
    Output('feature-importance-table', 'children'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def update_feature_importance_table(n_submit, n_clicks, client_id):
    if client_id is None:
        return "Please enter a valid client ID."

    feature_importance = get_client_feature_importance(client_id)

    if feature_importance is None:
        return "Failed to retrieve feature importance data."

    # Convert the feature_importance data into a DataFrame
    feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    table = dash_table.DataTable(
        columns=[{"name": "Feature", "id": "Feature"},
                 {"name": "Importance", "id": "Importance"}],
        data=feature_importance_df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    )

    return table

@app.callback(
    Output('negative-impact-features', 'children'),
    [Input('input-id', 'n_submit'), Input('submit-button', 'n_clicks')],
    State('input-id', 'value')
)
def display_negative_features(n_submit, n_clicks, client_id):
    if client_id is None:
        return "Veuillez entrer un ID client valide."

    feature_importance = get_client_feature_importance(client_id)

    if feature_importance is None:
        return "Impossible de récupérer les données d'importance des caractéristiques."

    # Filtrer les caractéristiques avec un impact négatif
    negative_features = [key for key, value in feature_importance.items() if value > 0]

    # Obtenir les données spécifiques du client
    client_data = get_full_client_data(client_id)
    if client_data is None:
        return "ID client non trouvé."

    # Créer une liste d'éléments HTML pour afficher les caractéristiques et leurs valeurs spécifiques au client
    return [html.Div(f"{feature}: {client_data[feature]}") for feature in negative_features]


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
