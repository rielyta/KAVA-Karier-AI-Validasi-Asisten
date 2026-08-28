# KAVA — Karier AI Validasi Asisten

> An AI-powered CV analysis platform for career prediction and skill gap recommendations.

🔗 **Live Demo:** [https://kava-karier-ai-validasi-asisten.vercel.app](https://kava-karier-ai-validasi-asisten.vercel.app/)

🔗 **Backend API:** [valiant-victory-production-032b.up.railway.app](https://valiant-victory-production-032b.up.railway.app)

🔗 **Streamlit Dashboard:** [kava-analytics.streamlit.app](https://kava-analytics.streamlit.app/)

🔗 **Streamlit GitHub Repo:** [github.com/clrsahlim/KAVA-Streamlit](https://github.com/clrsahlim/KAVA-Streamlit)

🔗 **Hugging Face AI Engine:** [https://huggingface.co/spaces/parulls/kava-ai-engine](https://huggingface.co/spaces/parulls/kava-ai-engine)

🔗 **Model Artifacts:** [AI Model Link](https://drive.google.com/drive/folders/1vpD8p66A0bX8fvA4HtVyovODLdJMRlaJ)

---

## Table of Contents

- [About KAVA](#about-kava)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Running Locally](#running-locally)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Team](#team)

---

## About KAVA

KAVA (Karier AI Validasi Asisten) is a web application that helps users automatically analyze their CVs using artificial intelligence. Users simply upload their CV in PDF format, and KAVA will:

1. Extract information from the CV (skills, experience, education, certifications)
2. Predict the **Top 3 Roles** most suited to the user's profile
3. Analyze the **skill gap** between the user's profile and industry requirements
4. Provide **personalized career advice** using Generative AI (Mistral)

---

## Features

- Register and log in with email/password or Google OAuth
- Email verification via OTP
- Forgot password and reset password via email
- Upload CV (PDF, max 5MB) and automatic analysis
- CV analysis history with pagination
- Profile management (update name, change/set password, delete account)
- Responsive layout for mobile and desktop

---

## Architecture

```
User (Browser)
      │
      ▼
Frontend — Vite + React (Railway)
      │  REST API (HTTPS)
      ▼
Backend API — Express.js (Railway)
      │
      ├──► CV Extractor — Flask + PyMuPDF (Railway)
      ├──► AI Model — TensorFlow + Mistral (Hugging Face)
      ├──► Email — Brevo HTTP API (Cloud)
      ├──► Google OAuth — google-auth-library (Cloud)
      └──► Database — PostgreSQL (Supabase)
```

---

## Tech Stack

| Component | Technology | Platform |
|---|---|---|
| Frontend | React.js + Vite + Tailwind CSS | Vercel |
| Backend API | Node.js + Express.js | Railway |
| CV Extractor | Python + Flask + PyMuPDF | Railway |
| AI Model | TensorFlow + Mistral API | Hugging Face |
| Database | PostgreSQL | Supabase |
| Email | Brevo HTTP API | Cloud |
| Google Auth | google-auth-library | Cloud |

---

## Repository Structure

```
KAVA-Karier-AI-Validasi-Asisten/
├── frontend/               # React + Vite
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── services/       # API calls
│   │   ├── hooks/
│   │   └── utils/
│   └── .env.example
│
├── backend/                # Express.js
│   ├── src/
│   │   ├── handlers/
│   │   ├── middleware/
│   │   ├── routes/
│   │   ├── services/
│   │   ├── utils/
│   │   └── server.js
│   ├── migrations/         # node-pg-migrate
│   └── .env.example
│
├── cv-extractor/           # Flask + PyMuPDF
│   ├── app.py
│   ├── extractor.py
│   ├── requirements.txt
│   └── Procfile
│
└── README.md
```

---

## Running Locally

### Prerequisites

- Node.js v18+
- Python 3.10+
- PostgreSQL
- Git

### 1. Clone the repository

```bash
git clone https://github.com/rielyta/KAVA-Karier-AI-Validasi-Asisten.git
cd KAVA-Karier-AI-Validasi-Asisten
```

### 2. Set up the database

Create a local PostgreSQL database:
```sql
CREATE DATABASE kava_db;
```

### 3. Set up the backend

```bash
cd backend
npm install
cp .env.example .env
# Fill in .env with your local configuration
npm run migrate
npm run start:dev
```

Backend runs at `http://localhost:5000`

### 4. Set up the CV extractor

```bash
cd cv-extractor
pip install -r requirements.txt
cp .env.example .env
python app.py
```

CV Extractor runs at `http://localhost:5001`

### 5. Set up the frontend

```bash
cd frontend
npm install
cp .env.example .env
# Fill in .env with your local configuration
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## Environment Variables

### Backend (`backend/.env`)

```env
HOST=localhost
PORT=5000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/kava_db

# JWT
ACCESS_TOKEN_KEY=your_access_token_secret
REFRESH_TOKEN_KEY=your_refresh_token_secret

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id

# Email (Brevo HTTP API)
BREVO_API_KEY=your_brevo_api_key
MAIL_FROM=your_email@gmail.com

# Services
CV_EXTRACTOR_URL=http://localhost:5001
AI_MODEL_URL=https://your-hf-space.hf.space
FRONTEND_URL=http://localhost:5173
```

### Frontend (`frontend/.env`)

```env
VITE_GOOGLE_CLIENT_ID=your_google_client_id
VITE_API_URL=http://localhost:5000
```

### CV Extractor (`cv-extractor/.env`)

```env
HOST=0.0.0.0
PORT=5001
FLASK_DEBUG=false
```

---

## API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/users/register` | Register a new account |
| POST | `/users/verify` | Verify email OTP |
| POST | `/users/resend-otp` | Resend OTP |
| POST | `/authentications/login` | Login with email/password |
| POST | `/authentications/google` | Login with Google OAuth |
| PUT | `/authentications/refresh` | Refresh access token |
| DELETE | `/authentications/logout` | Logout |

### User
| Method | Endpoint | Description |
|---|---|---|
| GET | `/users/me` | Get user profile |
| PUT | `/users/name` | Update name |
| PUT | `/users/password` | Update password (requires current password) |
| POST | `/users/password` | Set password (for Google accounts) |
| POST | `/users/forgot-password` | Request password reset link |
| POST | `/users/reset-password` | Reset password via token |
| DELETE | `/users/me` | Delete account |

### CV Analysis
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/cv/analyze` | Upload and analyze CV (form-data) |
| GET | `/api/cv/history` | CV analysis history |
| GET | `/api/cv/history/:id` | CV analysis detail |
| DELETE | `/api/cv/history/:id` | Delete analysis history |
| GET | `/api/cv/advice/:id` | Career advice from analysis |

> All user and CV analysis endpoints require the header: `Authorization: Bearer <accessToken>`

---

## Team

**CC26-PSU251 — Coding Camp powered by DBS Foundation 2026**

| Name | Role |
|---|---|
| Desi Maria Elita Silalahi | Full-Stack (Backend) |
| Nila Bi Idznillah | Full-Stack (Frontend) |
| Clara Angelin Pijoh | AI Engineer |
| Parulian Dwi Reslia Simbolon | AI Engineer |
| Ferarine Chang | Data Scientist |
| Clarissa Halim | Data Scientist |

---

> Built as a Capstone Project for Coding Camp powered by DBS Foundation 2026
