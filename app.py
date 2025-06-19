from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
scaler = joblib.load("scaler.pkl")
sleep_model = joblib.load("sleep_detection_model.pkl")
anomaly_model = joblib.load("anomaly_detection_model.pkl")

@app.route("/")  # Add this route
def index():
    return jsonify({"message": "Sleep Monitoring Flask Server is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        if features.shape[1] != 4:  # Validate input shape
            return jsonify({"error": "Expected 4 features: [HR, IBI_std, RR, motion]"}), 400
        scaled_features = scaler.transform(features)
        stage = sleep_model.predict(scaled_features)[0]
        anomaly_score = anomaly_model.decision_function(scaled_features)[0]
        is_anomaly = anomaly_score < -0.487123
        return jsonify({
            "stage": int(stage),
            "anomaly": bool(is_anomaly),
            "features": features.tolist()[0]
        })
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)