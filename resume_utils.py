# ============================================================
# ResumeGenie - resume_utils.py
#
# Part 1: Resume PDF extraction + analysis (process_resume)
# Part 2: ATS-friendly PDF rendering helpers
# ============================================================

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import fitz  # PyMuPDF


# ============================================================
# SECTION HEADING KEYWORDS
# ============================================================

SECTION_KEYWORDS = {
    "summary": [
        "summary", "professional summary", "profile",
        "objective", "career objective", "about me",
    ],
    "experience": [
        "experience", "work experience", "employment",
        "employment history", "work history",
        "professional experience", "internship", "internships",
    ],
    "education": [
        "education", "academic background", "academics",
        "educational qualification", "educational qualifications",
        "qualifications", "academic details",
    ],
    "projects": [
        "projects", "personal projects", "academic projects",
        "key projects", "selected projects", "project work",
    ],
    "skills": [
        "skills", "technical skills", "core skills",
        "key skills", "competencies", "technologies",
        "technical proficiencies", "skills & tools",
    ],
    "certifications": [
        "certifications", "certification", "certificates",
        "licenses", "courses", "courses & certifications",
        "training",
    ],
    "achievements": [
        "achievements", "accomplishments", "awards",
        "honors", "key achievements", "extra curricular",
        "extracurricular", "activities", "positions of responsibility",
    ],
}

BULLET_CHARS = ("\u2022", "\u25cf", "\u25aa", "\u2013", "\u2014", "-", "*", "o")


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from every page of a PDF using PyMuPDF.

    Uses "text" mode which preserves reading order well enough
    for single-column and most two-column resumes.
    """

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    try:
        pages = []

        for page in doc:
            page_text = page.get_text("text") or ""
            pages.append(page_text)

        full_text = "\n".join(pages)

    finally:
        doc.close()

    # Normalize whitespace but keep line structure
    full_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


# ============================================================
# CONTACT EXTRACTION
# ============================================================

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{4,6}"
)

LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/[A-Za-z0-9_\-/]+",
    re.IGNORECASE,
)

GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_\-]+",
    re.IGNORECASE,
)


def extract_contact(text: str) -> Dict[str, str]:
    """Pull contact details out of raw resume text."""

    contact = {
        "phone": "",
        "email": "",
        "linkedin": "",
        "github": "",
        "location": "",
    }

    email_match = EMAIL_RE.search(text)
    if email_match:
        contact["email"] = email_match.group(0)

    # Only look for a phone number in the first ~40 lines where
    # the header block normally lives, to avoid matching random
    # numbers buried in project descriptions.
    header_region = "\n".join(text.split("\n")[:40])

    phone_match = PHONE_RE.search(header_region)
    if phone_match:
        phone = re.sub(r"[^\d+]", "", phone_match.group(0))
        digits = re.sub(r"\D", "", phone)
        if 8 <= len(digits) <= 15:
            contact["phone"] = phone_match.group(0).strip()

    linkedin_match = LINKEDIN_RE.search(text)
    if linkedin_match:
        contact["linkedin"] = linkedin_match.group(0)

    github_match = GITHUB_RE.search(text)
    if github_match:
        contact["github"] = github_match.group(0)

    # Location: look for "City, State/Country" style patterns
    # near the email/phone line in the header.
    location_match = re.search(
        r"^([A-Z][A-Za-z .]+,\s*[A-Z][A-Za-z .]+)$",
        header_region,
        re.MULTILINE,
    )
    if location_match:
        candidate = location_match.group(1).strip()
        if len(candidate) <= 60:
            contact["location"] = candidate

    return contact


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into named sections by detecting
    common section headings on their own lines.

    Returns {section_name: section_text}.
    """

    lines = text.split("\n")

    # Map: line index -> canonical section name
    heading_positions: List[tuple[int, str]] = []

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line or len(line) > 60:
            continue

        normalized = re.sub(r"[^a-z &/]", "", line.lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)

        for section, keywords in SECTION_KEYWORDS.items():
            if normalized in keywords:
                heading_positions.append((index, section))
                break

    sections: Dict[str, str] = {}

    for pos_index, (line_index, section) in enumerate(heading_positions):
        start = line_index + 1

        if pos_index + 1 < len(heading_positions):
            end = heading_positions[pos_index + 1][0]
        else:
            end = len(lines)

        body = "\n".join(lines[start:end]).strip()

        # Keep the longest capture if a section is detected twice
        if section not in sections or len(body) > len(sections[section]):
            sections[section] = body

    return sections


# ============================================================
# METRICS
# ============================================================

def compute_metrics(text: str) -> Dict[str, Any]:
    """Compute objective resume metrics used by the scorer."""

    lines = [line for line in text.split("\n") if line.strip()]
    words = re.findall(r"[A-Za-z0-9'+/#.-]+", text)

    bullet_lines = [
        line for line in lines
        if line.strip().startswith(BULLET_CHARS)
    ]

    # A bullet counts as quantified if it contains a number,
    # percentage, or common magnitude marker.
    quantified = [
        line for line in bullet_lines
        if re.search(r"\d", line)
        or re.search(r"\b(percent|%|x\b|k\b|million|lakh|thousand)", line, re.IGNORECASE)
    ]

    bullet_count = len(bullet_lines)
    quantified_count = len(quantified)

    return {
        "word_count": len(words),
        "line_count": len(lines),
        "bullet_count": bullet_count,
        "quantified_bullet_count": quantified_count,
        "quantified_bullet_percentage": (
            round(quantified_count / bullet_count * 100, 1)
            if bullet_count else 0.0
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_resume(text: str, sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Decide whether the extracted text actually looks like a resume
    and is complete enough to analyze.
    """

    word_count = len(re.findall(r"[A-Za-z0-9'+/#.-]+", text))

    if word_count < 40:
        return {
            "is_valid": False,
            "message": (
                "Very little text could be extracted from this PDF. "
                "It may be a scanned image. Please upload a "
                "text-based (non-scanned) PDF resume."
            ),
        }

    found = set(sections.keys())

    core_sections = {"education", "experience", "projects", "skills"}

    if not (found & core_sections):
        return {
            "is_valid": False,
            "message": (
                "This document does not look like a resume - no "
                "standard sections (education, experience, projects, "
                "skills) were detected."
            ),
        }

    missing = []

    if "skills" not in found:
        missing.append("skills")

    if "education" not in found:
        missing.append("education")

    if not ({"experience", "projects"} & found):
        missing.append("experience or projects")

    message = "Resume looks good."

    if missing:
        message = (
            "Resume processed, but these sections appear to be "
            "missing: " + ", ".join(missing) + "."
        )

    return {
        "is_valid": True,
        "message": message,
        "missing_sections": missing,
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def process_resume(pdf_path: str) -> Dict[str, Any]:
    """
    Full non-LLM resume processing pipeline:

        PDF -> text -> contact + sections + metrics -> validation

    Returns a dict consumed by main.py and the LangGraph nodes.
    """

    try:
        text = extract_text_from_pdf(pdf_path)
    except ValueError as e:
        return {
            "success": False,
            "validation": {"is_valid": False, "message": str(e)},
            "text": "",
            "contact": {},
            "metrics": {},
            "sections": {},
        }

    sections = detect_sections(text)
    contact = extract_contact(text)
    metrics = compute_metrics(text)
    validation = validate_resume(text, sections)

    return {
        "success": bool(validation["is_valid"]),
        "validation": validation,
        "text": text,
        "contact": contact,
        "metrics": metrics,
        "sections": sections,
    }



# ============================================================
# PART 2: ATS-FRIENDLY PDF RENDERER
# ============================================================

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    KeepTogether,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# FONT SETUP
# ============================================================

def _register_fonts() -> tuple[str, str, str]:
    """
    Try to use common DejaVu fonts if available.
    Fall back to Helvetica if they are not installed.
    """

    regular_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    italic_name = "Helvetica-Oblique"

    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ),
    ]

    for regular, bold, italic in candidates:
        if os.path.exists(regular) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(
                    TTFont("ResumeGenie-Regular", regular)
                )
                pdfmetrics.registerFont(
                    TTFont("ResumeGenie-Bold", bold)
                )

                if os.path.exists(italic):
                    pdfmetrics.registerFont(
                        TTFont("ResumeGenie-Italic", italic)
                    )
                else:
                    pdfmetrics.registerFont(
                        TTFont("ResumeGenie-Italic", regular)
                    )

                return (
                    "ResumeGenie-Regular",
                    "ResumeGenie-Bold",
                    "ResumeGenie-Italic",
                )
            except Exception:
                pass

    return (
        regular_name,
        bold_name,
        italic_name,
    )


REGULAR_FONT, BOLD_FONT, ITALIC_FONT = _register_fonts()


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    """Convert arbitrary values to clean printable text."""

    if value is None:
        return ""

    text = str(value)

    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"[ \t]+", " ", text).strip()


def escape(text: Any) -> str:
    """
    Escape text for ReportLab Paragraph XML.
    """

    text = clean_text(text)

    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def join_nonempty(*values: Any, separator: str = " | ") -> str:
    parts = []

    for value in values:
        value = clean_text(value)

        if value:
            parts.append(value)

    return separator.join(parts)


def get_first(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)

        if value not in (None, ""):
            return value

    return ""


# ============================================================
# RESUME DATA NORMALIZATION
# ============================================================

def normalize_resume_data(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize the structured output produced by resume_builder_node.

    Expected structure:

    {
        "resume": {
            "name": "",
            "headline": "",
            "contact": {},
            "summary": "",
            "education": [],
            "experience": [],
            "projects": [],
            "skills": {},
            "certifications": [],
            "achievements": []
        }
    }

    The function is intentionally tolerant so minor LLM schema
    variations do not break PDF generation.
    """

    if not isinstance(analysis, dict):
        raise ValueError(
            "Resume builder output must be a dictionary."
        )

    resume = analysis.get(
        "resume",
        analysis
    )

    if not isinstance(resume, dict):
        raise ValueError(
            "Resume data is not a valid dictionary."
        )

    return {
        "name": clean_text(
            get_first(
                resume,
                "name",
                "full_name",
                "candidate_name",
            )
        ),

        "headline": clean_text(
            get_first(
                resume,
                "headline",
                "title",
                "professional_title",
            )
        ),

        "contact": resume.get(
            "contact",
            {}
        ),

        "summary": clean_text(
            get_first(
                resume,
                "summary",
                "professional_summary",
            )
        ),

        "education": normalize_list(
            resume.get(
                "education",
                []
            )
        ),

        "experience": normalize_list(
            resume.get(
                "experience",
                resume.get(
                    "work_experience",
                    []
                )
            )
        ),

        "projects": normalize_list(
            resume.get(
                "projects",
                []
            )
        ),

        "skills": normalize_skills(
            resume.get(
                "skills",
                {}
            )
        ),

        "certifications": normalize_list(
            resume.get(
                "certifications",
                []
            )
        ),

        "achievements": normalize_list(
            resume.get(
                "achievements",
                []
            )
        ),
    }


def normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def normalize_skills(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {
            "Technical Skills": [clean_text(value)]
            if value
            else []
        }

    result = {}

    for category, items in value.items():

        category_name = clean_text(
            category
        )

        if isinstance(items, list):
            cleaned_items = [
                clean_text(item)
                for item in items
                if clean_text(item)
            ]

        else:
            cleaned_items = [
                clean_text(items)
            ] if clean_text(items) else []

        if category_name and cleaned_items:
            result[
                category_name
            ] = cleaned_items

    return result


# ============================================================
# REPORTLAB STYLES
# ============================================================

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ResumeName",
            fontName=BOLD_FONT,
            fontSize=21,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeHeadline",
            fontName=REGULAR_FONT,
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeContact",
            fontName=REGULAR_FONT,
            fontSize=8.2,
            leading=10,
            alignment=TA_CENTER,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeSection",
            fontName=BOLD_FONT,
            fontSize=10.5,
            leading=12,
            spaceBefore=7,
            spaceAfter=3,
            textTransform="uppercase",
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeBody",
            fontName=REGULAR_FONT,
            fontSize=8.7,
            leading=11.2,
            spaceAfter=2.5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeBullet",
            fontName=REGULAR_FONT,
            fontSize=8.5,
            leading=10.8,
            leftIndent=11,
            firstLineIndent=-7,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeEntryTitle",
            fontName=BOLD_FONT,
            fontSize=9.2,
            leading=11,
            spaceAfter=1,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeEntryMeta",
            fontName=ITALIC_FONT,
            fontSize=8,
            leading=10,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ResumeSkill",
            fontName=REGULAR_FONT,
            fontSize=8.5,
            leading=11,
            spaceAfter=2,
        )
    )

    return styles


# ============================================================
# PDF ELEMENT HELPERS
# ============================================================

def section_heading(
    title: str,
    styles
):
    return [
        Spacer(1, 3),
        Paragraph(
            escape(title),
            styles["ResumeSection"]
        ),
        HRFlowable(
            width="100%",
            thickness=0.65,
            color=colors.black,
            spaceBefore=0,
            spaceAfter=4,
        ),
    ]


def bullet(
    text: Any,
    styles
):
    text = clean_text(text)

    if not text:
        return None

    return Paragraph(
        f"- {escape(text)}",
        styles["ResumeBullet"]
    )


def render_bullets(
    bullets: Any,
    styles
):
    elements = []

    if not isinstance(bullets, list):
        bullets = [bullets]

    for item in bullets:

        if isinstance(item, dict):

            text = get_first(
                item,
                "text",
                "description",
                "bullet",
                "action",
            )

        else:
            text = item

        element = bullet(
            text,
            styles
        )

        if element:
            elements.append(element)

    return elements


def render_entry(
    item: Any,
    styles,
    show_bullets: bool = True
):
    elements = []

    if isinstance(item, str):

        elements.append(
            bullet(
                item,
                styles
            )
        )

        return [
            element
            for element in elements
            if element is not None
        ]

    if not isinstance(item, dict):

        elements.append(
            bullet(
                str(item),
                styles
            )
        )

        return [
            element
            for element in elements
            if element is not None
        ]

    title = get_first(
        item,
        "title",
        "name",
        "project",
        "degree",
        "role",
        "position",
    )

    organization = get_first(
        item,
        "organization",
        "company",
        "institution",
        "university",
    )

    location = get_first(
        item,
        "location",
        "city",
    )

    dates = get_first(
        item,
        "dates",
        "date",
        "duration",
        "period",
    )

    meta = join_nonempty(
        organization,
        location,
        dates,
        separator=" | "
    )

    if title:

        elements.append(
            Paragraph(
                escape(title),
                styles["ResumeEntryTitle"]
            )
        )

    if meta:

        elements.append(
            Paragraph(
                escape(meta),
                styles["ResumeEntryMeta"]
            )
        )

    description = get_first(
        item,
        "description",
        "summary",
    )

    if description:

        elements.append(
            Paragraph(
                escape(description),
                styles["ResumeBody"]
            )
        )

    if show_bullets:

        bullets = get_first(
            item,
            "bullets",
            "responsibilities",
            "highlights",
            "achievements",
        )

        if bullets:
            elements.extend(
                render_bullets(
                    bullets,
                    styles
                )
            )

    link = get_first(
        item,
        "link",
        "url",
        "github",
        "demo",
    )

    if link:

        elements.append(
            Paragraph(
                escape(link),
                styles["ResumeBody"]
            )
        )

    elements.append(
        Spacer(1, 2)
    )

    return [
        element
        for element in elements
        if element is not None
    ]


# ============================================================
# MAIN PDF GENERATOR
# ============================================================

def generate_resume_pdf(
    analysis: Dict[str, Any],
    output_path: str,
    pagesize=A4,
) -> str:
    """
    Generate a clean ATS-friendly text PDF from AI-generated
    structured resume data.

    Returns the exact output path.
    """

    resume = normalize_resume_data(
        analysis
    )

    styles = build_styles()

    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_path)
        ),
        exist_ok=True
    )

    document = SimpleDocTemplate(
        output_path,
        pagesize=pagesize,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        title=resume["name"]
        or "ATS Resume",
        author="ResumeGenie",
        subject="ATS-friendly resume",
    )

    story = []

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    if resume["name"]:

        story.append(
            Paragraph(
                escape(resume["name"]),
                styles["ResumeName"]
            )
        )

    if resume["headline"]:

        story.append(
            Paragraph(
                escape(resume["headline"]),
                styles["ResumeHeadline"]
            )
        )

    contact = resume["contact"]

    if isinstance(contact, dict):

        contact_values = []

        for key in [
            "phone",
            "email",
            "linkedin",
            "github",
            "portfolio",
            "location",
        ]:

            value = contact.get(key)

            if value:
                contact_values.append(
                    clean_text(value)
                )

        if contact_values:

            story.append(
                Paragraph(
                    escape(
                        " | ".join(
                            contact_values
                        )
                    ),
                    styles["ResumeContact"]
                )
            )

    elif contact:

        story.append(
            Paragraph(
                escape(contact),
                styles["ResumeContact"]
            )
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if resume["summary"]:

        story.extend(
            section_heading(
                "Summary",
                styles
            )
        )

        story.append(
            Paragraph(
                escape(
                    resume["summary"]
                ),
                styles["ResumeBody"]
            )
        )

    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    if resume["education"]:

        story.extend(
            section_heading(
                "Education",
                styles
            )
        )

        for item in resume["education"]:

            story.extend(
                render_entry(
                    item,
                    styles,
                    show_bullets=False
                )
            )

    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    if resume["experience"]:

        story.extend(
            section_heading(
                "Experience",
                styles
            )
        )

        for item in resume["experience"]:

            story.extend(
                render_entry(
                    item,
                    styles
                )
            )

    # --------------------------------------------------------
    # Projects
    # --------------------------------------------------------

    if resume["projects"]:

        story.extend(
            section_heading(
                "Projects",
                styles
            )
        )

        for item in resume["projects"]:

            story.extend(
                render_entry(
                    item,
                    styles
                )
            )

    # --------------------------------------------------------
    # Skills
    # --------------------------------------------------------

    if resume["skills"]:

        story.extend(
            section_heading(
                "Technical Skills",
                styles
            )
        )

        for category, skills in resume[
            "skills"
        ].items():

            skill_text = join_nonempty(
                category + ":",
                ", ".join(skills),
                separator=" "
            )

            story.append(
                Paragraph(
                    escape(skill_text),
                    styles["ResumeSkill"]
                )
            )

    # --------------------------------------------------------
    # Certifications
    # --------------------------------------------------------

    if resume["certifications"]:

        story.extend(
            section_heading(
                "Certifications",
                styles
            )
        )

        for item in resume[
            "certifications"
        ]:

            if isinstance(item, dict):

                name = get_first(
                    item,
                    "name",
                    "title",
                    "certification",
                )

                issuer = get_first(
                    item,
                    "issuer",
                    "organization",
                    "provider",
                )

                date = get_first(
                    item,
                    "date",
                    "issued",
                )

                text = join_nonempty(
                    name,
                    issuer,
                    date,
                    separator=" | "
                )

            else:

                text = str(item)

            element = bullet(
                text,
                styles
            )

            if element:
                story.append(element)

    # --------------------------------------------------------
    # Achievements
    # --------------------------------------------------------

    if resume["achievements"]:

        story.extend(
            section_heading(
                "Achievements",
                styles
            )
        )

        for item in resume[
            "achievements"
        ]:

            if isinstance(item, dict):

                text = join_nonempty(
                    get_first(
                        item,
                        "title",
                        "name",
                        "achievement",
                    ),
                    get_first(
                        item,
                        "description",
                        "result",
                    ),
                    separator=" - "
                )

            else:

                text = str(item)

            element = bullet(
                text,
                styles
            )

            if element:
                story.append(element)

    # --------------------------------------------------------
    # Build PDF
    # --------------------------------------------------------

    if not story:

        raise ValueError(
            "No resume content was available "
            "for PDF generation."
        )

    document.build(
        story
    )

    return output_path


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def build_resume_pdf(
    analysis: Dict[str, Any],
    output_path: str,
    pagesize=A4,
) -> str:
    """
    Alias used by main.py if desired.
    """

    return generate_resume_pdf(
        analysis=analysis,
        output_path=output_path,
        pagesize=pagesize,
    )
