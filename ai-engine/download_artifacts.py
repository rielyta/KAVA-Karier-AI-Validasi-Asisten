import gdown, os

os.makedirs("artifacts", exist_ok=True)

ARTIFACTS = {
    "artifacts/kava_best_model.keras": "1VhSPhE2EfozO_F4QMJQEhKOCU5ucUexU",
    "artifacts/vectorizer.pkl":        "1vLKujoyLTPf4uTvWipS2J2KT5JRgRx1n",
    "artifacts/scaler.pkl":            "1xlK1-mHqfJ3LDW5yKtc9T-U-69_IoJ5e",
    "artifacts/label_encoder.pkl":     "1l4qozelHZVrv-TIHaM7s1HDjwfEOrc2u",
    "artifacts/skill_gap_db.json":     "1QPvn2mAomwBpNY0Iia_6HloI5ZX0kbkY",
}

for path, file_id in ARTIFACTS.items():
    if not os.path.exists(path):
        print(f"⬇️  Downloading {path}...")
        gdown.download(
            "https://drive.google.com/uc?id=1gGOa9SY0oN8L8Br1hWIxZJscQNGe93Le",
            path, quiet=False
        )
    else:
        print(f"✅ {path} sudah ada, skip.")