# EMNIST Handwriting Recognizer (Digits + Letters)

A complete project: train a CNN on EMNIST (digits 0-9, A-Z, a-z), then deploy
it as a web app where users draw a character and get the prediction.

## 1. Setup

```bash
pip install -r requirements.txt --break-system-packages
```

## 2. Train the model

```bash
python train_model.py
```

- Downloads EMNIST "byclass" dataset automatically (~1.7GB, first run only).
- Trains for 10 epochs (~10-30 min on GPU, longer on CPU).
- Saves the best model as `emnist_cnn.pth`.
- Expect ~85-88% test accuracy (EMNIST byclass is hard — many letters/digits
  look alike, e.g. '0'/'O', '1'/'l'/'I', '5'/'S').

To train faster for testing, reduce `EPOCHS` in `train_model.py` to 2-3.

## 3. Move the model into the app

```bash
cp emnist_cnn.pth app/
```

## 4. Run the web app

```bash
cd app
python app.py
```

Open **http://localhost:5000** — draw a digit or letter on the canvas and
click "Recognize".

## 5. Deploy (options)

### Quick: Render / Railway / PythonAnywhere
1. Push the `app/` folder (with `emnist_cnn.pth` inside) to a GitHub repo.
2. Add a `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Add `gunicorn` to requirements.txt.
4. Connect the repo on Render/Railway and deploy — they auto-detect Flask.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app/ /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt gunicorn
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```
```bash
docker build -t emnist-app .
docker run -p 5000:5000 emnist-app
```

### Hugging Face Spaces (Gradio alternative)
If you'd rather skip Flask, this same model can be wrapped in a Gradio
`sketchpad` interface and deployed free on HF Spaces — ask if you want that
version too.

## Project structure
```
emnist_project/
├── train_model.py     # CNN training script
├──continue_training.py 
├── export_model.py    # optional ONNX export
├── requirements.txt
├── README.md
└── app/
    ├── app.py          # Flask backend (loads emnist_cnn.pth)
    └── templates/
        └── index.html  # drawing canvas UI
```

## Notes
- Canvas captures a white background with dark strokes; `app.py` inverts
  colors and resizes to 28x28 to match EMNIST's format automatically.
- The model returns top-5 predictions with confidence scores, useful since
  similar-looking characters (0/O, 1/l, 5/S, 2/Z) are common confusions.
