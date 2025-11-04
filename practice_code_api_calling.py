import requests
import pandas as pd
url="http://127.0.0.1:5000/decisiontree_model_predict"
payload = {"my_input":[{
             "alcohol": 12.4,
            "malic_acid": 1.98,
            "ash": 2.20,
            "alcalinity_of_ash": 18.0,
            "magnesium": 96,
            "total_phenols": 2.10,
            "flavanoids": 1.89,
            "nonflavanoid_phenols": 0.27,
            "proanthocyanins": 1.04,
            "color_intensity": 4.28,
            "hue": 1.07,
            "od280/od315_of_diluted_wines": 2.85,
            "proline": 590
        }]}
response = requests.post(url,json=payload)
print(response.status_code)
print(response.json())