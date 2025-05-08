from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the saved GRU model
model = load_model("gru_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        input_seq = np.array(data['input'])  # shape: (10, num_features)
        input_seq = np.expand_dims(input_seq, axis=0)  # shape: (1, 10, num_features)

        prediction = model.predict(input_seq)[0][0]
        result = "High Risk" if prediction > 0.5 else "Low Risk"

        return jsonify({
            "prediction": result,
            "probability": float(prediction)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
