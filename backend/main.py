# backend/main.py
import os
import sys
import shutil
import time
from pathlib import Path

# --- 1. SETUP PATHS & IMPORTS ---
# Add the current directory (backend) to system path
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
MODELS_DIR = BACKEND_DIR / "models"

sys.path.append(str(BACKEND_DIR))

# --- STANDARD IMPORTS ---
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

# --- LOCAL MODULE IMPORTS ---
# Handles running from root vs inside folder
try:
    from database import SessionLocal, create_db_and_tables, User
    from security import get_password_hash
    from feature_extraction import extract_polygons_from_image, find_features
    from geometry_engine import build_3d_model
except ImportError:
    from backend.database import SessionLocal, create_db_and_tables, User
    from backend.security import get_password_hash
    from backend.feature_extraction import extract_polygons_from_image, find_features
    from backend.geometry_engine import build_3d_model

# --- 2. CONFIGURE GEMINI ---
# IMPORTANT: Set 'GEMINI_API_KEY' in Render Environment Variables!
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
# Public backend URL (Render)
BACKEND_PUBLIC_URL = os.environ.get("BACKEND_PUBLIC_URL", "")

# Initialize FastAPI
app = FastAPI(title="Blueprint 2 3D")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- 3. MOUNTS & TEMPLATES ---
# Specific mounts for organization
app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")
templates = Jinja2Templates(directory=FRONTEND_DIR)

# --- STARTUP ---
@app.on_event("startup")
def on_startup():
    print("Creating database and tables...")
    create_db_and_tables()

# --- AUTH MODELS ---
class UserCreate(BaseModel):
    email: str
    password: str

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- AI HELPER ---
def get_gemini_analysis(image_path_str):
    if not client: return "AI Analysis Unavailable: API Key missing."
    try:
        with open(image_path_str, "rb") as f: image_bytes = f.read()
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', 
            contents=["Analyze this floor plan. Identify rooms and flow.", types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "AI Analysis unavailable."

# ==========================================
#  API ROUTES
# ==========================================

@app.post("/convert")
async def convert_blueprint_to_3d(
    request: Request,
    blueprint_file: UploadFile = File(...),
    wall_height: float = Form(3.0),
    wall_thickness: int = Form(5)
):
    upload_path = UPLOAD_DIR / blueprint_file.filename
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(blueprint_file.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error_message": str(e)})

    model_filename = f"{Path(blueprint_file.filename).stem}_{int(time.time())}.obj"
    output_model_path = MODELS_DIR / model_filename
    model_url = f"{BACKEND_PUBLIC_URL}/models/{model_filename}" if BACKEND_PUBLIC_URL else f"/models/{model_filename}"


    try:
        # A. Gemini Analysis
        ai_analysis_text = get_gemini_analysis(str(upload_path))

        # B. 3D Conversion
        wall_polygons = extract_polygons_from_image(str(upload_path))
        if not wall_polygons:
            return JSONResponse(status_code=400, content={"success": False, "error_message": "No wall features detected."})

        # Locate Templates
        door_path = BACKEND_DIR / "door_template.png"
        win_path = BACKEND_DIR / "window_template.png"

        door_polygons = find_features(str(upload_path), str(door_path), 0.7) if door_path.exists() else []
        window_polygons = find_features(str(upload_path), str(win_path), 0.7) if win_path.exists() else []
        
        final_model = build_3d_model(
            wall_polygons, door_polygons, window_polygons, wall_height, wall_thickness
        )
        final_model.export(str(output_model_path))
        
        return JSONResponse(content={
            "success": True, "model_url": model_url, "ai_analysis": ai_analysis_text
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error_message": str(e)})

@app.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return JSONResponse(content={"success": True, "message": "User created!"})

# ==========================================
#  PAGE ROUTES
# ==========================================

@app.get("/health")
def health():
    return {"status": "ok"}


