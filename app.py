from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Murali, your Hugging Face Space is working!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

# import joblib
# from huggingface_hub import hf_hub_download
# from flask import Flask, request, jsonify
# import pandas as pd

# app = Flask(__name__)

# # Load model from Hugging Face Hub
# model_path = hf_hub_download(repo_id="Murali0606/tourism-model", filename="tourism_best_model.pkl")
# model = joblib.load(model_path)

# @app.route("/predict", methods=["POST"])
# def predict():
#     input_data = request.json
#     df = pd.DataFrame([input_data])
#     prediction = model.predict(df)[0]
#     return jsonify({"ProdTaken": int(prediction)})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=7860)
