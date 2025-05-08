import requests
import json

# Replace with your actual test input — should be shape (10, num_features), e.g., 10x57
# Make sure this matches the number of features used in your trained model!
test_input = {
    "input": [
        [0.2] * 57,
        [0.3] * 57,
        [0.4] * 57,
        [0.5] * 57,
        [0.6] * 57,
        [0.7] * 57,
        [0.8] * 57,
        [0.9] * 57,
        [1.0] * 57,
        [1.1] * 57
    ]
}

response = requests.post("http://127.0.0.1:5000/predict", json=test_input)
print("Response from API:")
print(response.json())

