import pytest
from api.api2 import app
from api.api2 import loaded_pipe_clf_params, df_wk, loaded_model_pipeline, explainer  


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(scope="module")
def global_data():
    return {
        'loaded_pipe_clf_params': loaded_pipe_clf_params,
        'df_wk': df_wk,
        'loaded_model_pipeline': loaded_model_pipeline,
        'explainer': explainer
    }

def test_model_loading(global_data):
    assert global_data['loaded_pipe_clf_params'] is not None
    assert 'pipeline' in global_data['loaded_pipe_clf_params']
    assert hasattr(global_data['loaded_model_pipeline'], 'predict')

def test_data_loading(global_data):
    assert global_data['df_wk'] is not None
    assert 'SK_ID_CURR' in global_data['df_wk'].columns

def test_data_transformation(global_data):
    sample_client_data = global_data['df_wk'].sample(1)

    print("Sample client data columns before dropping:", sample_client_data.columns)

    # Supprimer la colonne SK_ID_CURR si elle existe
    if 'SK_ID_CURR' in sample_client_data.columns:
        sample_client_data = sample_client_data.drop(['SK_ID_CURR'], axis=1)

    # Supprimer la colonne Pas_de_credit_en_cours si elle existe
    if 'Pas_de_credit_en_cours' in sample_client_data.columns:
        sample_client_data = sample_client_data.drop(['Pas_de_credit_en_cours'], axis=1)

    print("Sample client data columns after dropping:", sample_client_data.columns)

    transformed_data = sample_client_data

    n_features_in_model = global_data['loaded_model_pipeline'].named_steps['Classifier'].n_features_in_

    print("Number of features expected by the model:", n_features_in_model)

    assert transformed_data.shape[1] == n_features_in_model


def test_model_prediction(global_data):
    sample_client_data = global_data['df_wk'].sample(1)
    
    # Supprimer les colonnes inutiles si elles existent
    columns_to_remove = ['SK_ID_CURR', 'Pas_de_credit_en_cours']
    for col in columns_to_remove:
        if col in sample_client_data.columns:
            sample_client_data = sample_client_data.drop([col], axis=1)

    prediction = global_data['loaded_model_pipeline'].predict(sample_client_data)
    
    # Assertions
    assert prediction in [0, 1], f"Unexpected prediction value: {prediction}"

def test_lime_explainer_creation(global_data):
    assert global_data['explainer'] is not None
    assert hasattr(global_data['explainer'], 'explain_instance')

def test_full_client_data(client):
    response = client.get('/full_client_data?client_id=100003')  
    assert response.status_code == 200
    data = response.get_json()
    assert "AGE" in data

def test_prediction(client):
    response = client.get('/prediction?client_id=100003') 
    assert response.status_code == 200
    data = response.get_json()
    assert "probability" in data

def test_full_client_data_invalid_id(client):
    response = client.get('/full_client_data?client_id=invalid_id')
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Invalid Client ID"

