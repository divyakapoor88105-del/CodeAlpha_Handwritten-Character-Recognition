from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import re

app = Flask(__name__)

# -------------------------
# -------------------------
NUM_CLASSES = 62

class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def label_to_char(label):
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(label - 10 + ord('A'))
    else:
        return chr(label - 36 + ord('a'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMNIST_CNN(NUM_CLASSES).to(device)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location=device))
model.eval()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    # Strip data URL header (data:image/png;base64,...)
    img_str = re.sub("^data:image/.+;base64,", "", data)
    img_bytes = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize to 28x28 (EMNIST size)
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype(np.float32)

    # Canvas: white background, black strokes -> EMNIST: black bg, white strokes
    arr = 255.0 - arr
    arr = arr.T
    # Normalize same as training
    arr = (arr / 255.0 - 0.5) / 0.5

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,28,28]

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5)

    results = []
    for p, idx in zip(top5_probs[0], top5_idx[0]):
        results.append({
            "char": label_to_char(idx.item()),
            "confidence": round(p.item() * 100, 2)
        })

    return jsonify({"predictions": results})


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
