# deployment.py
from flask import Flask, request, jsonify
from utils import load_and_predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model_path = data.get("model_path", "trained_model.pkl")
    input_data = data.get("input_data", [])
    model_type = data.get("model_type", "ml")
    
    try:
        prediction = load_and_predict(model_path, input_data, model_type)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
