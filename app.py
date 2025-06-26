import os
import io
import requests
from PIL import Image
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from collections import OrderedDict

# === CONFIGURATION ===
MODEL_PATH = "best_vqa_model.pth"

# === INITIALISATION ===
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL DEFINITION ===
class VQAModel(nn.Module):
    def __init__(self, num_classes):
        super(VQAModel, self).__init__()
        self.image_encoder = models.mobilenet_v2(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bert = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.fc = nn.Sequential(
            nn.Linear(1280 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        with torch.no_grad():
            img_feat = self.image_encoder(image)
            img_feat = self.pool(img_feat).view(image.size(0), -1)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = bert_output.last_hidden_state[:, 0, :]
        combined = torch.cat((img_feat, txt_feat), dim=1)
        out = self.fc(combined)
        return out

# === LOAD TOKENIZER, TRANSFORMS, MODEL ===
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Charger le checkpoint et les métadonnées
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_labels = checkpoint['num_labels']
LABELS = list(checkpoint['label_map'].keys())

# Créer et charger le modèle avec renommage des clés
model = VQAModel(num_classes=num_labels).to(device)
raw_state = checkpoint['model_state_dict']
new_state = OrderedDict()
for k, v in raw_state.items():
    if k.startswith("mobilenet.features"):
        new_key = k.replace("mobilenet.features", "image_encoder")
    else:
        new_key = k
    new_state[new_key] = v
model.load_state_dict(new_state)
model.eval()

# === API ROUTE ===
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("image_url")
    question = data.get("question")

    if not image_url or not question:
        return jsonify({"error": "Missing image_url or question"}), 400

    try:
        # Load image
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Tokenize question
        encoded = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=64)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            logits = model(image_tensor, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            predicted_label = LABELS[pred_idx]

        return jsonify({
            "question": question,
            "answer": predicted_label,
            "confidence": round(probs[0][pred_idx].item(), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === START SERVER ===
if __name__ == "__main__":
    app.run(debug=True)
