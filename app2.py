from flask import Flask, request, jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
import io
import torch

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle Hugging Face
model_name = "dandelin/vilt-b32-finetuned-vqa"
model = ViltForQuestionAnswering.from_pretrained(model_name).to(device)
processor = ViltProcessor.from_pretrained(model_name)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")
    question = data.get("question")

    if not image_url or not question:
        return jsonify({"error": "Missing 'image_url' or 'question'."}), 400

    try:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        encoding = processor(image, question, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            idx = outputs.logits.argmax(-1).item()
            answer = model.config.id2label[idx]

        return jsonify({
            "question": question,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)