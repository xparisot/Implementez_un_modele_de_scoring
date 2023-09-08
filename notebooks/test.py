import requests as rq
from datetime import datetime

BASE_URL = 'http://Parisot.pythonanywhere.com'

payload = {'input': 'Ecriver quelque chose ici'}
response = rq.get(BASE_URL, params=payload)

# Check the response status code
if response.status_code == 200:
    try:
        json_values = response.json()
        rq_input, timestamp, character_count = json_values['input'], json_values['timestamp'], json_values['character_count']

        print(f'Input is: {rq_input}')
        print(f'Date is: {datetime.fromtimestamp(timestamp)}')
        print(f'Letter count is: {character_count}')
    except rq.exceptions.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.content)
