import requests

from ica_dummies.config import ICA_TOKEN, ICA_BASE_URL

URL = "https://api.nextgen-beta.ica.ibm.com/ica/v1/chat-models"

headers = {
    "accept": "application/json",
    "Authorization": f"{ICA_TOKEN}",
}

print("Request URL:", URL)

response = requests.get(URL, headers=headers, timeout=30)

print("Status Code:", response.status_code)

try:
    print(response.json())
except ValueError:
    print(response.text)

response.raise_for_status()
