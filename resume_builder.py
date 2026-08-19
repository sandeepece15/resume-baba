# ============================================================
# ResumeGenie - resume_builder.py
#
# Pipeline:
#   1. Groq rewrites the resume into a clean, ATS-friendly
#      JSON structure, applying any feedback supplied.
#   2. A SEPARATE provider — Mistral, falling back to
#      OpenRouter, then Groq — independently fact-checks that
#      draft against the ORIGINAL resume text. Returns a
#      corrected JSON version.
#   3. Deterministic backfill: name/contact details extracted
#      by PyMuPDF are force-merged in so nothing gets dropped.
#   4. The corrected JSON is rendered into a downloadable PDF.
# ============================================================

import json
import os
import re
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from graph import call_llm, clean_json_response


# ============================================================
# FONT SETUP
# ============================================================

def _register_fonts() -> tuple[str, str, str]:
    """
    Register DejaVu fonts when available (they cover bullets,
    dashes and accented characters). Fall back to Helvetica.
    """

    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ),
        (
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf",
        ),
    ]

    for regular, bold, italic in candidates:
        if os.path.exists(regular) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("RG-Regular", regular))
                pdfmetrics.registerFont(TTFont("RG-Bold", bold))
                pdfmetrics.registerFont(
                    TTFont("RG-Italic", italic if os.path.exists(italic) else regular)
                )
                return "RG-Regular", "RG-Bold", "RG-Italic"
            except Exception:
                pass

    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


REGULAR_FONT, BOLD_FONT, ITALIC_FONT = _register_fonts()


# ============================================================
# TEXT SAFETY (critical: ReportLab Paragraphs parse XML)
# ============================================================

def _clean(value: Any) -> str:
    """Normalize whitespace and typographic characters."""

    if value is None:
        return ""

    text = str(value)

    for old, new in {
        "–": "-", "—": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', " ": " ", "…": "...",
    }.items():
        text = text.replace(old, new)

    return re.sub(r"[ \t]+", " ", text).strip()


def esc(value: Any) -> str:
    """
    Escape text for ReportLab's XML-based Paragraph markup.

    Without this, a resume containing "Electronics & Communication"
    or "C/C++ & Python" crashes the whole PDF build.
    """

    text = _clean(value)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================
# FACT-CHECK PROVIDER FALLBACK
# ============================================================

def _fact_check_with_fallback(
    system_prompt: str,
    user_prompt: str,
) -> Optional[str]:
    """
    Run the fact-check pass on an independent provider chain:
    Mistral -> OpenRouter -> Groq. Returns None if every
    provider fails, so the caller can fall back to the draft.
    """

    for provider in ("mistral", "openrouter", "groq"):
        try:
            return call_llm(provider, system_prompt, user_prompt)
        except Exception:
            continue

    return None


# ============================================================
# JSON SCHEMA (shared by both prompts)
# ============================================================

RESUME_JSON_SCHEMA = """
{
  "name": "",
  "title": "",
  "contact": {
    "phone": "",
    "email": "",
    "linkedin": "",
    "github": "",
    "location": ""
  },
  "summary": "",
  "education": [
    {
      "degree": "",
      "institution": "",
      "location": "",
      "dates": "",
      "details": ""
    }
  ],
  "experience": [
    {
      "role": "",
      "company": "",
      "location": "",
      "dates": "",
      "bullets": []
    }
  ],
  "projects": [
    {
      "name": "",
      "link": "",
      "dates": "",
      "bullets": []
    }
  ],
  "skills": {
    "Category Name": ["skill 1", "skill 2"]
  },
  "certifications": [],
  "achievements": []
}
"""


# ============================================================
# PROMPTS
# ============================================================

BUILDER_SYSTEM_PROMPT = f"""
You are an elite resume writer and ATS (Applicant Tracking System)
optimization specialist. You rewrite resumes so they parse perfectly
in ATS software AND impress human recruiters.

You will receive the candidate's ORIGINAL resume text, structured
data already extracted from it, optionally a target job description,
and optionally feedback from a prior review to apply.

=====================================================
ABSOLUTE RULES — NEVER BREAK THESE
=====================================================
1. NEVER invent companies, job titles, dates, metrics, schools,
   certifications, or skills that are not in the original resume.
2. You MAY rephrase, restructure, strengthen verbs, reorder bullets,
   fix grammar, and improve clarity.
3. Only include a number/metric if it already exists in the original
   text or is a literal count of something explicitly present.
4. If a section has no content in the original (e.g. no work
   experience), omit that section — do not fabricate.
5. Never use first-person pronouns (I, me, my).
6. Copy the candidate's name and contact details EXACTLY as given
   in the extracted contact info. Never alter them.

=====================================================
ATS WRITING RULES — APPLY TO EVERY SECTION
=====================================================

TITLE:
- Write a targeted professional title matching the candidate's
  actual profile (e.g. "Aspiring Data Analyst", "Computer Science
  Graduate"). If a job description is provided, align the title
  with it ONLY if the candidate's background genuinely supports it.

SUMMARY (2-3 sentences, max 60 words):
- Formula: [who they are] + [strongest skills/technologies] +
  [what they want / value they bring].
- Pack in the most important keywords from the job description
  that the candidate genuinely has.
- No fluff like "hard-working team player".

BULLETS (experience & projects):
- Formula: strong action verb + what was built/done + tools used +
  quantified outcome when one exists in the original.
- Start every bullet with a varied action verb (Built, Developed,
  Engineered, Automated, Analyzed, Designed, Implemented, Optimized,
  Visualized, Reduced, Improved). Never start two consecutive
  bullets with the same verb.
- Keep each bullet to ONE line of thought, 12-25 words.
- Put the most impressive, most job-relevant bullet first in
  each entry.
- Use past tense for completed work, present tense only for
  ongoing roles.
- Preserve every real metric from the original (scores, %,
  dataset sizes, user counts) and surface them near the start
  of the bullet where natural.

SKILLS:
- Group into 3-6 ATS-standard categories such as "Programming
  Languages", "Data Analysis", "Data Visualization", "Tools &
  Platforms", "Databases", "Frameworks".
- Only include skills actually present in the original resume.
- If a job description is provided, reorder skills so the most
  JD-relevant ones appear first within their category.

EDUCATION:
- Keep degree, institution, location, dates, and scores/grades
  exactly as in the original. Put relevant coursework in details.

KEYWORD ALIGNMENT (only when a job description is provided):
- Mirror the JD's exact terminology where the candidate's real
  experience supports it (e.g. if the JD says "data pipelines"
  and the candidate built data pipelines, use that phrase).
- Never stuff in keywords the candidate cannot back up.

=====================================================
OUTPUT FORMAT
=====================================================
You MUST return ONLY valid JSON, with exactly this structure:

{RESUME_JSON_SCHEMA}

Return ONLY JSON. No markdown, no commentary, no code fences.
"""


REVIEW_SYSTEM_PROMPT = f"""
You are a strict resume fact-checker and ATS auditor.

You will be given the candidate's ORIGINAL resume text and a
REWRITTEN resume draft in JSON.

Check the draft against the original and verify:
1. No fact has been fabricated: no new companies, titles, dates,
   metrics, schools, certifications, or skills that don't genuinely
   exist in the original resume.
2. Nothing important from the original was silently dropped —
   especially contact details, education entries, projects,
   certifications, and real metrics.
3. Bullets follow ATS style: action-verb led, concise, no
   first-person pronouns.
4. The JSON structure is complete and every field has the
   correct type.

If the draft is accurate, return it back UNCHANGED as JSON.

If you find fabricated, exaggerated, or unsupported content, REMOVE
or CORRECT it in place so it only reflects what the original resume
actually supports, restore anything important that was dropped,
and return the corrected JSON.

You MUST return ONLY valid JSON, with exactly this structure:

{RESUME_JSON_SCHEMA}

Return ONLY JSON. No markdown, no commentary, no code fences.
"""


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    return [
        _clean(item)
        for item in value
        if _clean(item)
    ]


def _normalize_entry(entry: Any, bullet_key: str = "bullets") -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None

    entry = {k: _clean(v) if isinstance(v, str) else v for k, v in entry.items()}

    if bullet_key in entry:
        value = entry.get(bullet_key)

        if bullet_key == "details":
            # details is free text (coursework, grades), not a bullet
            # list — a list value gets joined, a string is kept as-is.
            if isinstance(value, list):
                entry[bullet_key] = "; ".join(
                    _clean(v) for v in value if _clean(v)
                )
            else:
                entry[bullet_key] = _clean(value)
        else:
            entry[bullet_key] = _normalize_str_list(value)

    return entry


def normalize_resume_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarantee a predictable structure for the PDF renderer,
    regardless of small variations in what the LLM returns.
    """

    if not isinstance(data, dict):
        raise ValueError("Resume builder did not return a JSON object.")

    contact = data.get("contact", {})
    if not isinstance(contact, dict):
        contact = {}

    education_raw = data.get("education", [])
    experience_raw = data.get("experience", [])
    projects_raw = data.get("projects", [])
    skills_raw = data.get("skills", {})

    education = [
        e for e in (
            _normalize_entry(item, "details")
            for item in (education_raw if isinstance(education_raw, list) else [])
        )
        if e
    ]

    experience = [
        e for e in (
            _normalize_entry(item)
            for item in (experience_raw if isinstance(experience_raw, list) else [])
        )
        if e
    ]

    projects = [
        e for e in (
            _normalize_entry(item)
            for item in (projects_raw if isinstance(projects_raw, list) else [])
        )
        if e
    ]

    if isinstance(skills_raw, dict):
        skills = {
            _clean(category): _normalize_str_list(items)
            for category, items in skills_raw.items()
            if _normalize_str_list(items)
        }
    else:
        skills = {}

    return {
        "name": _clean(data.get("name", "")),
        "title": _clean(data.get("title", "")),
        "contact": {
            "phone": _clean(contact.get("phone", "")),
            "email": _clean(contact.get("email", "")),
            "linkedin": _clean(contact.get("linkedin", "")),
            "github": _clean(contact.get("github", "")),
            "location": _clean(contact.get("location", "")),
        },
        "summary": _clean(data.get("summary", "")),
        "education": education,
        "experience": experience,
        "projects": projects,
        "skills": skills,
        "certifications": _normalize_str_list(data.get("certifications", [])),
        "achievements": _normalize_str_list(data.get("achievements", [])),
    }


# ============================================================
# DETERMINISTIC BACKFILL
# ============================================================

def _backfill_identity(
    resume: Dict[str, Any],
    resume_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    The LLM occasionally drops or mangles identity fields.
    Force-merge the deterministically extracted contact info
    (from PyMuPDF) back in — extracted data wins for contact,
    since it came straight from the original PDF.
    """

    extracted = resume_data.get("contact", {}) or {}
    text = resume_data.get("text", "") or ""

    contact = resume.setdefault("contact", {})

    for key in ("phone", "email", "linkedin", "github", "location"):
        extracted_value = _clean(extracted.get(key, ""))
        if extracted_value:
            contact[key] = extracted_value

    # Name fallback: first non-empty line of the original resume
    # that looks like a person's name (short, no digits, no @).
    if not resume.get("name"):
        for line in text.split("\n")[:10]:
            candidate = _clean(line)
            if (
                candidate
                and len(candidate) <= 50
                and not re.search(r"[\d@|]", candidate)
                and len(candidate.split()) <= 5
            ):
                resume["name"] = candidate
                break

    return resume


# ============================================================
# GENERATION PIPELINE
# ============================================================

def build_improved_resume(
    resume_text: str,
    resume_data: Dict[str, Any],
    job_description: str = "",
    feedback: str = "",
) -> Dict[str, Any]:
    """
    Two-provider pipeline:

      1. Draft: Groq rewrites the resume into ATS-friendly JSON.
      2. Fact-check: an independent provider chain
         (Mistral -> OpenRouter -> Groq) audits the draft against
         the original resume text.
      3. Deterministic backfill of identity fields.
    """

    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    contact = resume_data.get("contact", {})
    sections = resume_data.get("sections", {})

    draft_prompt = f"""
ORIGINAL RESUME TEXT
============================================================
{resume_text}

EXTRACTED CONTACT INFO (copy these EXACTLY)
============================================================
{json.dumps(contact)}

EXTRACTED SECTIONS (already detected)
============================================================
{json.dumps(sections)}

TARGET JOB DESCRIPTION
============================================================
{job_description.strip() if job_description.strip() else "Not provided"}

FEEDBACK TO APPLY (from a prior resume review, if any)
============================================================
{feedback.strip() if feedback.strip() else "None provided - apply general ATS best practices."}

Rewrite this resume now, following every rule in the system prompt.
"""

    raw_draft = call_llm(
        "groq",
        BUILDER_SYSTEM_PROMPT,
        draft_prompt
    )

    try:
        draft_json = json.loads(
            clean_json_response(raw_draft)
        )
    except Exception as e:
        raise ValueError(
            f"Resume builder returned invalid JSON: {e}"
        )

    # ----------------------------------------------------
    # Fact-check pass on an independent provider chain.
    # Any failure here falls back to the valid first draft
    # rather than failing the whole request.
    # ----------------------------------------------------

    review_prompt = f"""
ORIGINAL RESUME TEXT
============================================================
{resume_text}

REWRITTEN DRAFT (JSON)
============================================================
{json.dumps(draft_json)}

Audit the draft now, following every rule in the system prompt.
"""

    reviewed_json = draft_json

    raw_reviewed = _fact_check_with_fallback(
        REVIEW_SYSTEM_PROMPT,
        review_prompt
    )

    if raw_reviewed:
        try:
            candidate = json.loads(
                clean_json_response(raw_reviewed)
            )
            # Only accept the audit if it returned real content
            if isinstance(candidate, dict) and (
                candidate.get("name")
                or candidate.get("education")
                or candidate.get("projects")
                or candidate.get("experience")
            ):
                reviewed_json = candidate
        except Exception:
            pass

    normalized = normalize_resume_json(reviewed_json)

    return _backfill_identity(normalized, resume_data)


# ============================================================
# PDF RENDERING
# ============================================================
#
# ATS rendering principles applied here:
#   - single column, no tables for layout of body content
#     (one exception: entry header rows use a borderless 2-cell
#     table purely for right-aligned dates, which all major
#     ATS parsers handle fine)
#   - standard fonts, real text (selectable/copyable)
#   - every string XML-escaped so '&', '<', '>' never crash
#   - standard section headings in an ATS-optimal order
#   - PDF metadata set (title/author) for parser identification
# ============================================================

def _build_styles() -> Dict[str, ParagraphStyle]:

    return {
        "name": ParagraphStyle(
            "Name",
            fontName=BOLD_FONT,
            fontSize=20,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111111"),
            spaceAfter=2,
        ),
        "title": ParagraphStyle(
            "Headline",
            fontName=REGULAR_FONT,
            fontSize=10.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=3,
        ),
        "contact": ParagraphStyle(
            "Contact",
            fontName=REGULAR_FONT,
            fontSize=8.6,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName=BOLD_FONT,
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#111111"),
            spaceBefore=9,
            spaceAfter=2,
        ),
        "entry_left": ParagraphStyle(
            "EntryLeft",
            fontName=BOLD_FONT,
            fontSize=9.8,
            leading=12,
            textColor=colors.HexColor("#111111"),
        ),
        "entry_right": ParagraphStyle(
            "EntryRight",
            fontName=REGULAR_FONT,
            fontSize=8.8,
            leading=12,
            alignment=TA_RIGHT,
            textColor=colors.HexColor("#444444"),
        ),
        "entry_sub": ParagraphStyle(
            "EntrySub",
            fontName=ITALIC_FONT,
            fontSize=8.8,
            leading=11,
            textColor=colors.HexColor("#444444"),
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName=REGULAR_FONT,
            fontSize=9.2,
            leading=12.4,
            textColor=colors.HexColor("#1a1a1a"),
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=REGULAR_FONT,
            fontSize=9.2,
            leading=12.4,
            textColor=colors.HexColor("#1a1a1a"),
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=2,
        ),
        "skill": ParagraphStyle(
            "Skill",
            fontName=REGULAR_FONT,
            fontSize=9.2,
            leading=12.6,
            textColor=colors.HexColor("#1a1a1a"),
            spaceAfter=2,
        ),
    }


def _section_header(text: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    return [
        Paragraph(esc(text).upper(), styles["section"]),
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#111111"),
            spaceBefore=0,
            spaceAfter=4,
        ),
    ]


def _bullet_list(items: List[str], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    elements = []

    for item in items:
        text = _clean(item)
        if text:
            elements.append(
                Paragraph(f"&bull;&nbsp;{esc(text)}", styles["bullet"])
            )

    return elements


def _entry_header(
    left_text: str,
    right_text: str,
    styles: Dict[str, ParagraphStyle],
    available_width: float,
) -> Any:
    """
    Entry title on the left, dates right-aligned — the layout
    recruiters expect, and safe for ATS parsers.
    """

    left = Paragraph(esc(left_text), styles["entry_left"])
    right = Paragraph(esc(right_text), styles["entry_right"])

    table = Table(
        [[left, right]],
        colWidths=[available_width * 0.72, available_width * 0.28],
    )

    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )

    return table


def render_resume_pdf(resume: Dict[str, Any], output_path: str) -> str:

    styles = _build_styles()

    page_width, _ = A4
    left_margin = right_margin = 0.6 * inch
    available_width = page_width - left_margin - right_margin

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=left_margin,
        rightMargin=right_margin,
        title=_clean(resume.get("name")) or "ATS Resume",
        author=_clean(resume.get("name")) or "ResumeGenie",
        subject="ATS-friendly resume",
        creator="ResumeGenie",
    )

    elements: List[Any] = []

    # ----------------------------------------------------
    # Header: name / title / contact
    # ----------------------------------------------------

    if resume.get("name"):
        elements.append(Paragraph(esc(resume["name"]), styles["name"]))

    if resume.get("title"):
        elements.append(Paragraph(esc(resume["title"]), styles["title"]))

    contact = resume.get("contact", {})

    contact_bits = [
        _clean(v)
        for v in [
            contact.get("phone"),
            contact.get("email"),
            contact.get("linkedin"),
            contact.get("github"),
            contact.get("location"),
        ]
        if _clean(v)
    ]

    if contact_bits:
        elements.append(
            Paragraph(esc(" | ".join(contact_bits)), styles["contact"])
        )

    elements.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#111111"),
            spaceBefore=4,
            spaceAfter=2,
        )
    )

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------

    if resume.get("summary"):
        elements.extend(_section_header("Professional Summary", styles))
        elements.append(Paragraph(esc(resume["summary"]), styles["body"]))

    # ----------------------------------------------------
    # Skills (high up = better ATS keyword matching)
    # ----------------------------------------------------

    if resume.get("skills"):
        elements.extend(_section_header("Skills", styles))

        for category, items in resume["skills"].items():
            if items:
                elements.append(
                    Paragraph(
                        f"<b>{esc(category)}:</b> {esc(', '.join(items))}",
                        styles["skill"],
                    )
                )

    # ----------------------------------------------------
    # Experience
    # ----------------------------------------------------

    if resume.get("experience"):
        elements.extend(_section_header("Experience", styles))

        for entry in resume["experience"]:
            block: List[Any] = []

            head = " - ".join(
                x for x in [entry.get("role"), entry.get("company")] if x
            )

            if head:
                block.append(
                    _entry_header(head, entry.get("dates", ""), styles, available_width)
                )

            sub = " | ".join(
                x for x in [entry.get("company") if entry.get("role") else "",
                             entry.get("location")] if x
            )
            if sub:
                block.append(Paragraph(esc(sub), styles["entry_sub"]))

            block.extend(_bullet_list(entry.get("bullets", []), styles))
            block.append(Spacer(1, 4))

            if block:
                elements.append(KeepTogether(block))

    # ----------------------------------------------------
    # Projects
    # ----------------------------------------------------

    if resume.get("projects"):
        elements.extend(_section_header("Projects", styles))

        for entry in resume["projects"]:
            block = []

            head = entry.get("name", "")
            if entry.get("link"):
                head = f"{head} | {entry['link']}" if head else entry["link"]

            if head:
                block.append(
                    _entry_header(head, entry.get("dates", ""), styles, available_width)
                )

            block.extend(_bullet_list(entry.get("bullets", []), styles))
            block.append(Spacer(1, 4))

            if block:
                elements.append(KeepTogether(block))

    # ----------------------------------------------------
    # Education
    # ----------------------------------------------------

    if resume.get("education"):
        elements.extend(_section_header("Education", styles))

        for entry in resume["education"]:
            block = []

            if entry.get("degree"):
                block.append(
                    _entry_header(
                        entry["degree"],
                        entry.get("dates", ""),
                        styles,
                        available_width,
                    )
                )

            sub = " | ".join(
                x for x in [entry.get("institution"), entry.get("location")] if x
            )
            if sub:
                block.append(Paragraph(esc(sub), styles["entry_sub"]))

            if entry.get("details"):
                block.append(Paragraph(esc(entry["details"]), styles["body"]))

            block.append(Spacer(1, 4))

            if block:
                elements.append(KeepTogether(block))

    # ----------------------------------------------------
    # Certifications
    # ----------------------------------------------------

    if resume.get("certifications"):
        elements.extend(_section_header("Certifications", styles))
        elements.extend(_bullet_list(resume["certifications"], styles))

    # ----------------------------------------------------
    # Achievements
    # ----------------------------------------------------

    if resume.get("achievements"):
        elements.extend(_section_header("Achievements", styles))
        elements.extend(_bullet_list(resume["achievements"], styles))

    # ----------------------------------------------------
    # Guard: never ship a near-blank PDF silently
    # ----------------------------------------------------

    if len(elements) <= 2:
        elements.append(Spacer(1, 12))
        elements.append(
            Paragraph(
                "No resume content was available to render.",
                styles["body"],
            )
        )

    doc.build(elements)

    return output_path
