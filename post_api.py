import requests
import json

from ml import EXAMPLE_CENSUS_DATA_POS

response = requests.post('https://still-forest-76228.herokuapp.com/predict/',
                         data=json.dumps(EXAMPLE_CENSUS_DATA_POS))

print(response.status_code)
print(response.json())
