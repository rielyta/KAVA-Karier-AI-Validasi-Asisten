import gdown, os

os.makedirs("artifacts", exist_ok=True)

ARTIFACTS = {
    "artifacts/kava_best_model.keras": "1Pdeu5rsx_X_yLgd1Lat7FV_ssZ6gsdir",
    "artifacts/vectorizer.pkl":        "10Tz68JctcT1gs581o-6ttuOZRBGIpQk5",
    "artifacts/scaler.pkl":            "1G6tTayHIlj-kfBguui9g29McThXnAcvw",
    "artifacts/label_encoder.pkl":     "1_vd6iq3WPFwmKhCQ_MXDtQe-NxRatrl_",
    "artifacts/skill_gap_db.json":     "1Pil7nnJB94vzol6pKP0x0sIHrNOhYM5Z",
    "artifacts/tfidf_vectorizer.pkl":  "108-5TWPP_S4n7UVzEDzu8dVV1A3TkTN1",
    "artifacts/domain_vectors.pkl":    "15UpC4pqIxI5LmaygnxpZ-MdjT_q3e-Oh",
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