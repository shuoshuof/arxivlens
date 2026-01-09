import requests
import os
import uuid

api_key = 'sk-TpmyAx3mIMmiivJ2tZulONiCg309yMKt91lmlm7XIF4'
url = "http://localhost:7863/api/v1/run/2e9e24cb-a2d2-41a1-93fd-aa0deda917ba"  # The complete API endpoint URL for this flow

# Request payload configuration
payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "Hello, how are you?"
}
payload["session_id"] = str(uuid.uuid4())

headers = {"x-api-key": api_key}

try:
    # Avoid routing localhost requests through HTTP proxy env vars.
    session = requests.Session()
    session.trust_env = False
    # Send API request
    response = session.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")
