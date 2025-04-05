import requests
import json

ABSOLUTE_IMAGE_PATH = "sample_cyst.jpg"

# the input image is expected to be preprocessed from here

API_URL = "http://localhost:8080/infer"
payload = {
    "image_path": ABSOLUTE_IMAGE_PATH
}

# Send POST request
response = requests.post(API_URL, json=payload)

# Handle response
if response.status_code == 200:
    result = response.json()
    print("Inference result:")
    print(json.dumps(result, indent=2))

    # add your other code here for post processing
else:
    print("Error:", response.status_code, response.text)
