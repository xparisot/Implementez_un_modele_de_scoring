import pytest
from api.api2 import app
from ..api2 import loaded_pipe_clf_params, df_wk, columns_for_prediction, loaded_model_pipeline, explainer

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
        'columns_for_prediction': columns_for_prediction,
        'loaded_model_pipeline': loaded_model_pipeline,
        'explainer': explainer
    }

def test_model_loading(global_data):
    assert global_data['loaded_pipe_clf_params'] is not None
    assert 'pipeline' in loaded_pipe_clf_params
    assert hasattr(loaded_model_pipeline, 'predict')

def test_data_loading():
    assert df_wk is not None
    assert 'SK_ID_CURR' in df_wk.columns

def test_data_transformation():
    sample_client_data = df_wk.sample(1).drop(['SK_ID_CURR'], axis=1)
    transformed_data = sample_client_data[columns_for_prediction]
    
    assert len(transformed_data.columns) == len(columns_for_prediction)   
    assert transformed_data.shape[1] == loaded_model_pipeline.named_steps['Classifier'].n_features_in_

def test_model_prediction():
    sample_client_data = df_wk.sample(1).drop(['SK_ID_CURR'], axis=1)
    input_data = sample_client_data[columns_for_prediction]
    
    prediction = loaded_model_pipeline.predict(input_data)
    prediction_proba = loaded_model_pipeline.predict_proba(input_data)
    
    assert prediction in [0, 1]
    assert 0 <= prediction_proba[0][0] <= 1
    assert 0 <= prediction_proba[0][1] <= 1

def test_lime_explainer_creation():
    assert explainer is not None
    assert hasattr(explainer, 'explain_instance')

def test_full_client_data(client):
    response = client.get('/full_client_data?client_id=123654')  
    assert response.status_code == 200
    data = response.get_json()
    assert "AGE" in data

def test_prediction(client):
    response = client.get('/prediction?client_id=123654') 
    assert response.status_code == 200
    data = response.get_json()
    assert "probability" in data

def test_full_client_data_invalid_id(client):
    response = client.get('/full_client_data?client_id=invalid_id')  
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Invalid Client ID"

