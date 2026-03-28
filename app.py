import joblib
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download

app = Flask(__name__)

model_path = hf_hub_download(repo_id="Murali0606/tourism-model", filename="tourism_best_model.pkl")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json
    # Convert dict values to list for prediction
    X = [list(input_data.values())]
    prediction = model.predict(X)[0]
    return jsonify({"ProdTaken": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)


