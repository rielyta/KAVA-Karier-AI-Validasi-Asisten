import gdown, os

os.makedirs("artifacts", exist_ok=True)

ARTIFACTS = {
    "artifacts/kava_best_model.keras": "1mP9Jh6JpGq-i7aTFosE5PVbBIkhjImqf",
    "artifacts/vectorizer.pkl":        "1Gwn0rQxd32Abg35F03rkmvobnOFu50Gr",
    "artifacts/scaler.pkl":            "1bIrXkMoPY_ibf0BB4Pb4_5zzWVX3SfcH",
    "artifacts/label_encoder.pkl":     "1HLNegQ0JWQ52kaH2iP_JxLpR1jcAqy15",
    "artifacts/skill_gap_db.json":     "1SsxqEmTgRbCMg_AAxlwdKsLIW0s1inqW",
    "artifacts/tfidf_vectorizer.pkl":  "1FVbY48uDLmN7rzZPy3HnOqd2s_Ztgw_D",
    "artifacts/domain_vectors.pkl":    "1B5zqjX72BtJOLEVnDYH2-eONnkj1P7ol",
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