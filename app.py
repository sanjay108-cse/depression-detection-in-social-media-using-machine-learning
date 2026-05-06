from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model      = pickle.load(open("model/model.pkl",      "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)


def convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


# ── DeepFace import with detailed error logging ──────────────────────────────
DEEPFACE_AVAILABLE = False
DEEPFACE_ERROR     = ""

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF C++ logs
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")           # suppress Python TF logs
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully.")
except Exception as e:
    DEEPFACE_ERROR = str(e)
    print(f"⚠ DeepFace failed to load: {e}")

# ── Emotion → depression weight mapping ──────────────────────────────────────
EMOTION_DEPRESSION_WEIGHTS = {
    "sad":      0.90,
    "fear":     0.70,
    "disgust":  0.50,
    "angry":    0.55,
    "contempt": 0.45,
    "neutral":  0.25,
    "surprise": 0.15,
    "happy":    0.05,
}

UPLOAD_DIR = "uploads"


def save_temp_image(img_b64: str) -> str:
    """Decode base64 image, save to disk, return file path."""
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    tmp_path = os.path.join(UPLOAD_DIR, "temp_face.jpg")
    img.save(tmp_path)
    return tmp_path


def analyze_face(tmp_path: str) -> dict:
    """Run DeepFace emotion analysis and return structured result."""
    result = DeepFace.analyze(
        img_path=tmp_path,
        actions=["emotion"],
        enforce_detection=False,
        silent=True
    )
    if isinstance(result, list):
        result = result[0]

    emotions = result.get("emotion", {})
    dominant = result.get("dominant_emotion", "neutral").lower()
    total    = sum(emotions.values()) or 1
    emo_pct  = {k: round(float(v) / total * 100, 1) for k, v in emotions.items()}

    dep_score = EMOTION_DEPRESSION_WEIGHTS.get(dominant, 0.2)
    is_dep    = dep_score > 0.5
    confidence = round(dep_score * 100, 1)

    return {
        "is_depressive":    is_dep,
        "prediction":       "Depressive" if is_dep else "Non-Depressive",
        "confidence":       confidence,
        "dominant_emotion": dominant,
        "emotions":         emo_pct,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    """Health-check endpoint — confirms DeepFace status."""
    return jsonify({
        "deepface_available": DEEPFACE_AVAILABLE,
        "deepface_error":     DEEPFACE_ERROR if not DEEPFACE_AVAILABLE else None,
    })


@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned    = clean_text(text)
    vector     = vectorizer.transform([cleaned])
    pred       = model.predict(vector)[0]
    proba      = model.predict_proba(vector)[0]
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({
        "is_depressive": bool(pred == 1),
        "prediction":    "Depressive" if pred == 1 else "Non-Depressive",
        "confidence":    confidence,
        "source":        "text"
    })


@app.route("/predict/face", methods=["POST"])
def predict_face():
    if not DEEPFACE_AVAILABLE:
        return jsonify({
            "error":   "DeepFace failed to load.",
            "details": DEEPFACE_ERROR
        }), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    tmp_path = None
    try:
        tmp_path = save_temp_image(img_b64)
        face_data = analyze_face(tmp_path)
        face_data["source"] = "face"
        return jsonify(convert_to_serializable(face_data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict/combined", methods=["POST"])
def predict_combined():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    text    = data.get("text", "").strip()
    img_b64 = data.get("image", "")
    results = {}

    # ── Text analysis ──────────────────────────────────────────────────────
    if text:
        cleaned   = clean_text(text)
        vector    = vectorizer.transform([cleaned])
        pred      = model.predict(vector)[0]
        proba     = model.predict_proba(vector)[0]
        text_conf = round(float(max(proba)) * 100, 1)
        results["text"] = {
            "is_depressive": bool(pred == 1),
            "confidence":    text_conf,
            "prediction":    "Depressive" if pred == 1 else "Non-Depressive",
        }

    # ── Face analysis ──────────────────────────────────────────────────────
    if img_b64:
        if not DEEPFACE_AVAILABLE:
            results["face_error"] = f"DeepFace unavailable: {DEEPFACE_ERROR}"
        else:
            tmp_path = None
            try:
                tmp_path  = save_temp_image(img_b64)
                face_data = analyze_face(tmp_path)
                results["face"] = convert_to_serializable(face_data)
            except Exception as e:
                results["face_error"] = str(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # ── Fusion ─────────────────────────────────────────────────────────────
    if "text" in results and "face" in results:
        t = results["text"]
        f = results["face"]
        fused_conf = round(t["confidence"] * 0.6 + f["confidence"] * 0.4, 1)
        fused_dep  = t["is_depressive"] or f["is_depressive"]
        results["fused"] = {
            "is_depressive": fused_dep,
            "confidence":    fused_conf,
            "prediction":    "Depressive" if fused_dep else "Non-Depressive",
        }

    if not results:
        return jsonify({"error": "No text or image provided"}), 400

    return jsonify(results)


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(debug=True, port=5000)