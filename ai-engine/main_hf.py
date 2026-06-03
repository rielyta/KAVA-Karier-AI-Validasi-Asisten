import os, json, pickle, re
import httpx
import uvicorn
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from sklearn.metrics.pairwise import cosine_similarity

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

# CUSTOM LAYER
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
        
    with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open(f"{ARTIFACT_DIR}/domain_vectors.pkl", "rb") as f:
        domain_vectors = pickle.load(f)

    assert set(domain_vectors.keys()) == set(label_encoder.classes_), "⚠️ ERROR: Key domain_vectors dan label_encoder tidak sinkron!"
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
    has_highlights: int = Field(default=0, ge=0, le=1, description="1 = punya highlights, 0 = tidak")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s,\.\'\-\+#/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# HELPER FUNCTIONS
def _prepare_features(data: CVPayload):
    has_cert = 1 if data.cert_count > 0 else 0
    has_hl   = data.has_highlights

    skills_count_est = 0 if not data.skills_raw.strip() else min(len([s for s in data.skills_raw.split(',') if s.strip()]), 36)

    num_feat = np.array([[
        min(data.experience_years, 40),
        skills_count_est,
        has_cert,
        data.cert_count,
        data.has_education,
        has_hl
    ]], dtype=np.float32)

    return data_scaler.transform(num_feat)


def _get_top3_predictions(text: str, num_scaled: np.ndarray, skills_raw: str) -> list:
    parts_dl = []
    if skills_raw.strip():
        for _ in range(4):
            parts_dl.append(skills_raw)
    parts_dl.append(text)
    
    cleaned_dl_text = clean_text(' '.join(parts_dl))
    text_token = text_vectorizer(np.array([cleaned_dl_text]))
    
    probs = kava_model.predict(
        {'text_input': text_token, 'numeric_input': num_scaled},
        verbose=0
    )[0]
    
    cleaned_tfidf_text = clean_text(f"{skills_raw} {text}")
    cv_vector = tfidf_vectorizer.transform([cleaned_tfidf_text])
    
    kw_scores = {}
    for domain, domain_vec in domain_vectors.items():
        sim_score = cosine_similarity(cv_vector, domain_vec)[0][0]
        kw_scores[domain] = round(sim_score * 100, 1)

    combined_scores = {}
    for i, role in enumerate(label_encoder.classes_):
        dl_score  = float(probs[i]) * 100
        semantic_score = kw_scores.get(role, 0.0) 
        
        combined_scores[role] = {
            'dl_confidence': round(dl_score, 2),
            'keyword_score': semantic_score,
            'final_score':   round(dl_score * 0.8 + semantic_score * 0.2, 2)
        }

    top3 = sorted(combined_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:3]
    return [
        {
            "rank":          rank + 1,
            "role":          role,
            "dl_confidence": scores["dl_confidence"],
            "keyword_score": scores["keyword_score"],
            "final_score":   scores["final_score"]
        }
        for rank, (role, scores) in enumerate(top3)
    ]
    
def _analyze_skill_gap(cv_skills_raw: str, full_cv_text: str, predicted_role: str) -> dict:
    cv_skills = {s.strip().lower() for s in cv_skills_raw.split(',') if s.strip()}
    required  = skill_gap_db.get(predicted_role, [])
    full_text_lower = f" {full_cv_text.lower()} " 

    matched = []
    missing = []
    
    for req in required:
        req_lower = req.lower()
    
        if req_lower in cv_skills:
            matched.append(req)
        elif f" {req_lower} " in full_text_lower or f" {req_lower}," in full_text_lower:
            matched.append(req)
        else:
            missing.append(req)

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
async def predict_career(data: CVPayload):
    try:
        num_scaled = _prepare_features(data)
        top3       = _get_top3_predictions(data.text, num_scaled, data.skills_raw)
        gap        = _analyze_skill_gap(data.skills_raw, data.text, top3[0]["role"])

        # ─── Mistral AI Career Coach ───
        prompt = f"""Anda adalah KAVA (Karier AI Validasi Asisten), asisten karir profesional.

        Profil Kandidat:
        - Role diprediksi (Top-1): {top3[0]['role']} ({top3[0]['final_score']}% final score)
        - Pengalaman: {data.experience_years} tahun
        - Skills yang dimiliki: {data.skills_raw}
        - Skills yang perlu ditingkatkan: {', '.join(gap['missing_skills'])}

        Berikan dalam Bahasa Indonesia:
        1. **Saran Karir**: Langkah konkret berikutnya untuk kandidat ini
        2. **Skill yang Harus Dipelajari**: Dari daftar missing skills, pilih 3 yang paling penting dan jelaskan mengapa
        3. **Rekomendasi Sertifikasi**: 1 sertifikasi internasional yang paling relevan untuk menembus pasar kerja {top3[0]['role']}

        Format respons: ringkas, actionable, maksimal 200 kata."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            mistral_resp = await client.post(
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
def skill_gap_only(role: str, skills_raw: str, text: str = ""):
    try:
        result = _analyze_skill_gap(skills_raw, text, role)
        return {"status": "success", "skill_gap": result}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
