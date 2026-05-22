import gdown, os

os.makedirs("artifacts", exist_ok=True)

ARTIFACTS = {
    "artifacts/kava_best_model.keras": "1f3-iaSIcv-s6Nl7mUzgRx1kRlS3zfjbV",
    "artifacts/vectorizer.pkl":        "1RWKRP1nRYvTdgJuq-yhf208UDtrrAi2E",
    "artifacts/scaler.pkl":            "1cM-cC0olab8_o9ujjHrlvA8CO70eJ2Ca",
    "artifacts/label_encoder.pkl":     "1F1aenFjluvUgIR9XLfvCAjhzH35DTK0A",
    "artifacts/skill_gap_db.json":     "1LvLaZcbJX-KakaS_JpqIoncs6u_CZGbI",
}

for path, file_id in ARTIFACTS.items():
    if not os.path.exists(path):
        print(f"⬇️  Downloading {path}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            path, quiet=False
        )
    else:
        print(f"✅ {path} sudah ada, skip.")