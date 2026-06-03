import gdown, os

os.makedirs("artifacts", exist_ok=True)

ARTIFACTS = {
    "artifacts/kava_best_model.keras": "1ql27D_QguVqhFSbWTCqdeYrNBIoQlv4K",
    "artifacts/vectorizer.pkl":        "12ESnbUSJ4De3SrvUm-dJihvsNe1Arb09",
    "artifacts/scaler.pkl":            "1FJmKOxcJPdJ1EQUcDIx41Lq-XJ3f81J6",
    "artifacts/label_encoder.pkl":     "1h_aBWlYW1r4NQ-idL4T8iGdes7GwTBfX",
    "artifacts/skill_gap_db.json":     "1hUHedXJGJWF2VhRnC1PYy1Tf25uG1Ak1",
    "artifacts/tfidf_vectorizer.pkl":  "1BRhGuJY0ktm0e-8g_QKTEa48tsyabo3R",
    "artifacts/domain_vectors.pkl":    "1qt4fJgKTw8KhDju6AKnT-kHw1594nTBP",
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