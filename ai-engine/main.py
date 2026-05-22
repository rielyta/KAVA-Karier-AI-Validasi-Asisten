import os, json, pickle
import uvicorn
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
from fastapi.responses import RedirectResponse

load_dotenv()

# KONFIGURASI MISTRAL API
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("⚠️ WARNING: MISTRAL_API_KEY tidak ditemukan di environment variables/file .env!")

# FASTAPI APP + CORS
app = FastAPI(
    title="KAVA AI Engine API",
    description="Prediksi Top-3 Role Karir & Analisis Skill Gap berbasis Deep Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CUSTOM LAYER (harus didefinisikan sebelum load_model)
@tf.keras.utils.register_keras_serializable()
class StripMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
    def call(self, inputs):
        return inputs
    def compute_mask(self, inputs, mask=None):
        return None
    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention = tf.keras.layers.Dense(units, activation='tanh')
        self.score     = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None, training=False):
        attn_weights = self.score(self.attention(inputs))
        if mask is not None:
            mask_f = tf.cast(tf.expand_dims(mask, -1), attn_weights.dtype)
            attn_weights += (1.0 - mask_f) * -1e9
        attn_weights = tf.nn.softmax(attn_weights, axis=1)
        return tf.reduce_sum(inputs * attn_weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, smoothing=0.1, name='focal_loss', **kwargs): 
        super().__init__(name=name, reduction='none', **kwargs)
        self.gamma     = gamma
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        n_cls  = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_oh   = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        y_oh   = y_oh * (1.0 - self.smoothing) + (self.smoothing / n_cls)
        ce     = -tf.reduce_sum(y_oh * tf.math.log(y_pred), axis=-1)
        p_cor  = tf.reduce_sum(y_oh * y_pred, axis=-1)
        return tf.pow(1.0 - p_cor, self.gamma) * ce

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'smoothing': self.smoothing})
        return config

# LOAD SEMUA ARTIFACT MODEL SAAT STARTUP
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

try:
    kava_model = tf.keras.models.load_model(
        f"{ARTIFACT_DIR}/kava_best_model.keras",
        custom_objects={
            'StripMask': StripMask,
            'AttentionPooling': AttentionPooling,
            'FocalLoss': FocalLoss
        }
    )

    with open(f"{ARTIFACT_DIR}/vectorizer.pkl", "rb") as f:
        vec_payload = pickle.load(f)
    text_vectorizer = tf.keras.layers.TextVectorization.from_config(vec_payload['config'])
    text_vectorizer.set_weights(vec_payload['weights'])

    with open(f"{ARTIFACT_DIR}/scaler.pkl", "rb") as f:
        data_scaler = pickle.load(f)

    with open(f"{ARTIFACT_DIR}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(f"{ARTIFACT_DIR}/skill_gap_db.json", "r") as f:
        skill_gap_db = json.load(f)

    print("✅ KAVA API siap — semua artifact berhasil dimuat.")

except Exception as e:
    print(f"❌ GAGAL MEMUAT MODEL: {e}")
    raise RuntimeError(f"Model load error: {e}")


# REQUEST/RESPONSE SCHEMAS
class CVPayload(BaseModel):
    text: str = Field(..., min_length=10, description="Teks gabungan dari CV")
    skills_raw: str = Field(..., description="Skills mentah dipisah koma")
    experience_years: float = Field(..., ge=0, le=40, description="Pengalaman dalam tahun (0-40)")
    cert_count: int = Field(..., ge=0, description="Jumlah sertifikasi")
    has_education: int = Field(..., ge=0, le=1, description="1 = punya info pendidikan, 0 = tidak")


# HELPER FUNCTIONS
def _prepare_features(data: CVPayload):
    is_fresh = 1 if data.experience_years < 2 else 0
    is_exp   = 1 if data.experience_years >= 5 else 0
    has_cert = 1 if data.cert_count > 0 else 0
    has_hl   = 1  # assume ada highlights jika ada summary

    skills_count_est = min(len(data.skills_raw.split(',')), 36)

    num_feat = np.array([[
        min(data.experience_years, 40),
        skills_count_est,
        has_cert,
        data.cert_count,
        data.has_education,
        is_fresh,
        is_exp,
        has_hl
    ]], dtype=np.float32)

    return data_scaler.transform(num_feat)


def _get_top3_predictions(text: str, num_scaled: np.ndarray) -> list:
    text_token = text_vectorizer(np.array([text]))
    probs      = kava_model.predict(
        {'text_input': text_token, 'numeric_input': num_scaled},
        verbose=0
    )[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    return [
        {
            "rank":       int(rank),
            "role":       label_encoder.inverse_transform([int(idx)])[0],
            "confidence": round(float(probs[idx]) * 100, 2)
        }
        for rank, idx in enumerate(top3_idx, 1)
    ]


def _analyze_skill_gap(cv_skills_raw: str, predicted_role: str) -> dict:
    cv_skills = set(s.strip().lower() for s in cv_skills_raw.split(',') if s.strip())
    required  = skill_gap_db.get(predicted_role, [])

    matched  = [s for s in required if any(s in cs or cs in s for cs in cv_skills)]
    missing  = [s for s in required if s not in matched]
    coverage = round(len(matched) / len(required) * 100, 1) if required else 0

    return {
        "required_skills":    required,
        "matched_skills":     matched,
        "missing_skills":     missing[:10],
        "coverage_pct":       coverage,
        "progress_bar_value": coverage / 100
    }


# ENDPOINTS

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")

@app.get("/api/health", summary="Health Check")
def health_check():
    return {"status": "ok", "service": "KAVA AI Engine v2.0"}


@app.post("/api/predict-career", summary="Top-3 Role Prediction + Skill Gap + AI Coach")
def predict_career(data: CVPayload):
    try:
        num_scaled = _prepare_features(data)
        top3       = _get_top3_predictions(data.text, num_scaled)
        gap        = _analyze_skill_gap(data.skills_raw, top3[0]["role"])

        # ─── Mistral AI Career Coach ───
        prompt = f"""Anda adalah KAVA (Karier AI Validasi Asisten), asisten karir profesional.

Profil Kandidat:
- Role diprediksi (Top-1): {top3[0]['role']} ({top3[0]['confidence']}% confidence)
- Pengalaman: {data.experience_years} tahun
- Skills yang dimiliki: {data.skills_raw}
- Skills yang perlu ditingkatkan: {', '.join(gap['missing_skills'])}

Berikan dalam Bahasa Indonesia:
1. **Saran Karir**: Langkah konkret berikutnya untuk kandidat ini
2. **Skill yang Harus Dipelajari**: Dari daftar missing skills, pilih 3 yang paling penting dan jelaskan mengapa
3. **Rekomendasi Sertifikasi**: 1 sertifikasi internasional yang paling relevan untuk menembus pasar kerja {top3[0]['role']}

Format respons: ringkas, actionable, maksimal 200 kata."""

        mistral_resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
                },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400
            },
            timeout=30
)
        mistral_resp.raise_for_status()
        coach_text = mistral_resp.json()["choices"][0]["message"]["content"]

        return {
            "status":           "success",
            "top3_role_matches": top3,
            "skill_gap":        gap,
            "ai_career_coach":  coach_text
        }

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error AI Engine: {str(err)}")


@app.post("/api/skill-gap", summary="Skill Gap saja (tanpa prediksi ulang)")
def skill_gap_only(role: str, skills_raw: str):
    try:
        result = _analyze_skill_gap(skills_raw, role)
        return {"status": "success", "skill_gap": result}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)