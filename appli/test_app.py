import pytest
from dash.testing.application_runners import import_app


@pytest.fixture
def dash_br():
    # Cette fixture est nécessaire pour lancer l'application et exécuter les tests
    with import_app("your_dash_app_filename_without_py_extension") as app:
        yield app


def test_client_info_table(dash_br):
    # Ce test vérifie si la table des informations du client est correctement rendue après avoir entré un ID client

    # Sélectionnez l'élément input et définissez une valeur d'ID client
    client_input = dash_br.find_element("#input-id")
    client_input.send_keys("12345")

    # Cliquez sur le bouton de soumission
    submit_button = dash_br.find_element("#submit-button")
    submit_button.click()

    # Vérifiez si la table d'informations du client est affichée
    table = dash_br.find_element("#client-info-table")
    assert table is not None


