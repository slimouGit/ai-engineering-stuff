# python
import os
import sys
import requests

from config import ICA_TOKEN

url = "https://api.nextgen-beta.ica.ibm.com/ica/v1/agents"

headers = {
    "accept": "application/json",
    "Authorization": ICA_TOKEN,
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print("Status Code:", response.status_code)
    try:
        print(response.json())
    except ValueError:
        print(response.text)
except requests.RequestException as e:
    print("Request failed:", e)