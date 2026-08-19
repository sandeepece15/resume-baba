import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from graph import run_resume_feature
from resume_builder import build_improved_resume, render_resume_pdf
from resume_utils import process_resume


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="ResumeGenie API",
    description="AI-powered resume analysis and career assistant",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FRONTEND_FILE = BASE_DIR / "frontend.html"

TEMP_DIR = BASE_DIR / "temp"

TEMP_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
async def home():
    """
    Serve the ResumeGenie frontend.
    """

    if FRONTEND_FILE.exists():
        return FileResponse(
            FRONTEND_FILE
        )

    return {
        "message": "ResumeGenie API is running."
    }


@app.get("/health")
async def health():
    """
    Health-check endpoint.
    """

    return {
        "status": "healthy",
        "service": "ResumeGenie"
    }


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

async def save_uploaded_pdf(
    file: UploadFile
) -> str:
    """
    Save uploaded PDF temporarily and return its path.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
        dir=TEMP_DIR
    )

    temp_path = temp_file.name

    try:
        with temp_file:
            shutil.copyfileobj(
                file.file,
                temp_file
            )

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise

    return temp_path


# ============================================================
# UPLOAD + PROCESS RESUME
# ============================================================

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...)
):
    """
    Upload a resume and extract its information.

    This endpoint does not call an LLM.

    It only performs:
        PDF
          ↓
        PyMuPDF
          ↓
        text extraction
          ↓
        basic processing
    """

    temp_path = None

    try:
        temp_path = await save_uploaded_pdf(
            file
        )

        resume_data = process_resume(
            temp_path
        )

        if not resume_data.get("success"):
            raise HTTPException(
                status_code=400,
                detail=resume_data.get(
                    "validation",
                    {}
                ).get(
                    "message",
                    "Unable to process resume."
                )
            )

        return {
            "success": True,
            "filename": file.filename,
            "validation": resume_data["validation"],
            "contact": resume_data["contact"],
            "metrics": resume_data["metrics"],
            "sections": resume_data["sections"],
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resume processing failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# MAIN RESUME ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    feature: str = Form(...),
    job_description: str = Form("")
):
    """
    Main ResumeGenie endpoint.

    Supported features:

        checker
        scorer
        career_coach
        cover_letter

    Flow:

        Upload PDF
             ↓
        PyMuPDF
             ↓
        Resume processing
             ↓
        LangGraph
             ↓
        Selected LLM
             ↓
        JSON response
    """

    temp_path = None

    try:

        # ----------------------------------------------------
        # Validate feature
        # ----------------------------------------------------

        allowed_features = {
            "checker",
            "resume_checker",
            "scorer",
            "resume_scorer",
            "career",
            "career_coach",
            "cover",
            "cover_letter",
        }

        feature = feature.lower().strip()

        if feature not in allowed_features:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid feature. Choose one of: "
                    "checker, scorer, career_coach, "
                    "cover_letter."
                )
            )

        # ----------------------------------------------------
        # Cover letter requires JD
        # ----------------------------------------------------

        if feature in {
            "cover",
            "cover_letter"
        } and not job_description.strip():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Job description is required "
                    "for cover letter generation."
                )
            )

        # ----------------------------------------------------
        # Save PDF
        # ----------------------------------------------------

        temp_path = await save_uploaded_pdf(
            file
        )

        # ----------------------------------------------------
        # Process resume
        # ----------------------------------------------------

        resume_data = process_resume(
            temp_path
        )

        if not resume_data.get("success"):
            raise HTTPException(
                status_code=400,
                detail=resume_data.get(
                    "validation",
                    {}
                ).get(
                    "message",
                    "Unable to read resume."
                )
            )

        resume_text = resume_data.get(
            "text",
            ""
        )

        # ----------------------------------------------------
        # Run LangGraph
        # ----------------------------------------------------

        result = run_resume_feature(
            feature=feature,
            resume_text=resume_text,
            job_description=job_description,
            resume_data=resume_data,
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,
            "feature": feature,
            "filename": file.filename,
            "result": result,
            "metrics": resume_data.get(
                "metrics",
                {}
            ),
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# CAREER COACH CHAT ENDPOINT
# ============================================================

@app.post("/career-coach")
async def career_coach(
    file: UploadFile = File(...),
    question: str = Form(...),
    job_description: str = Form("")
):
    """
    Career Coach endpoint.

    The question is added to the resume context so the
    career coach can answer based on the candidate's resume.
    """

    temp_path = None

    try:

        if not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        temp_path = await save_uploaded_pdf(
            file
        )

        resume_data = process_resume(
            temp_path
        )

        if not resume_data.get("success"):
            raise HTTPException(
                status_code=400,
                detail="Unable to process resume."
            )

        resume_text = resume_data["text"]

        # Add the user's question to the resume context.
        enhanced_job_description = f"""
Target Job Description:

{job_description if job_description.strip() else "Not provided"}

Candidate Question:

{question}
"""

        result = run_resume_feature(
            feature="career_coach",
            resume_text=resume_text,
            job_description=enhanced_job_description,
            resume_data=resume_data,
        )

        return {
            "success": True,
            "question": question,
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Career coach failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# BUILD ATS-FRIENDLY RESUME (PDF)
# ============================================================

@app.post("/build-resume")
async def build_resume(
    file: UploadFile = File(...),
    job_description: str = Form(""),
    feedback: str = Form("")
):
    """
    Generate a rewritten, ATS-friendly resume as a downloadable PDF.

    Flow:

        Upload PDF
             ↓
        PyMuPDF text extraction
             ↓
        Groq drafts an improved resume (structured JSON)
             ↓
        Groq self-checks the draft against the original
        (removes/corrects anything fabricated or dropped)
             ↓
        Render corrected JSON to PDF
             ↓
        Return the PDF file
    """

    temp_path = None
    output_pdf_path = None

    try:

        temp_path = await save_uploaded_pdf(
            file
        )

        resume_data = process_resume(
            temp_path
        )

        if not resume_data.get("success"):
            raise HTTPException(
                status_code=400,
                detail=resume_data.get(
                    "validation",
                    {}
                ).get(
                    "message",
                    "Unable to read resume."
                )
            )

        resume_text = resume_data.get(
            "text",
            ""
        )

        improved_resume = build_improved_resume(
            resume_text=resume_text,
            resume_data=resume_data,
            job_description=job_description,
            feedback=feedback,
        )

        safe_stem = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in Path(file.filename).stem
        )[:60] or "resume"

        output_filename = (
            f"improved_{safe_stem}_{uuid.uuid4().hex[:8]}.pdf"
        )

        output_pdf_path = TEMP_DIR / output_filename

        render_resume_pdf(
            improved_resume,
            str(output_pdf_path)
        )

        return FileResponse(
            path=str(output_pdf_path),
            filename="ATS_Friendly_Resume.pdf",
            media_type="application/pdf",
            background=BackgroundTask(
                lambda: output_pdf_path.unlink(missing_ok=True)
            ),
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Resume building failed: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )