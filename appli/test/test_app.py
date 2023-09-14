import pytest
import requests_mock
from dash_api import get_client_info_details, get_client_prediction, get_client_info, get_info_banque

def test_get_client_info_details(requests_mock):
    mock_response = {'key': 'value'}
    requests_mock.get('https://api2-398a526923d8.herokuapp.com/full_client_data?client_id=100002', json=mock_response)
    result = get_client_info_details(100002)
    assert result == mock_response

def test_get_client_prediction(requests_mock):
    mock_response = {'decision': 'accorde', 'probability': 90}
    requests_mock.get('https://api2-398a526923d8.herokuapp.com/prediction?client_id=100002', json=mock_response)
    decision, probability = get_client_prediction(100002)
    assert decision == 'accorde'
    assert probability == 90

def test_get_client_info(requests_mock):
    mock_response = {'key': 'value'}
    requests_mock.get('https://api2-398a526923d8.herokuapp.com/info_client?client_id=100002', json=mock_response)
    result = get_client_info(100002)
    assert result == mock_response

def test_get_info_banque(requests_mock):
    mock_response = {'key': 'value'}
    requests_mock.get('https://api2-398a526923d8.herokuapp.com/info_banque?client_id=100002', json=mock_response)
    result = get_info_banque(100002)
    assert result == mock_response
