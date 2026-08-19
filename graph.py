# ============================================================
# ResumeGenie - graph.py
# ============================================================

import json
import re
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from llm import get_llm


# ============================================================
# STATE
# ============================================================

class ResumeState(TypedDict, total=False):
    resume_text: str
    resume_data: Dict[str, Any]
    job_description: str
    feature: str
    result: Any
    error: str


# ============================================================
# PROMPTS
# ============================================================

CHECKER_SYSTEM_PROMPT = """
You are an expert resume reviewer and ATS specialist.

Analyze the candidate's resume carefully.

Focus on:
- ATS compatibility
- formatting
- missing sections
- weak content
- contact information
- project descriptions
- skills
- education
- experience
- measurable achievements
- readability

You MUST return ONLY valid JSON.

Do NOT return:
- Markdown
- code fences
- explanations outside JSON
- headings
- commentary

The JSON MUST follow exactly this structure:

{
  "overall_score": 0,
  "summary": "",
  "scores": {
    "ats": 0,
    "content": 0,
    "impact": 0,
    "keywords": 0,
    "structure": 0,
    "readability": 0
  },
  "categories": {
    "ats": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "content": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "impact": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "keywords": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "structure": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "readability": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    }
  },
  "strengths": [],
  "weaknesses": [],
  "improvements": []
}

Each issue found for a category MUST be an object:

{
  "title": "",
  "severity": "low",
  "evidence": "",
  "recommendation": ""
}

Each improvement MUST be an object:

{
  "priority": 1,
  "title": "",
  "why": "",
  "action": ""
}

SCORING:

90-100 = excellent
75-89  = good
60-74  = warning
0-59   = poor

Important:

- Evaluate the actual resume content.
- Student projects, hackathons, academic work and certifications
  are legitimate evidence.
- Do not penalize a student simply for having no formal employment.
- Do not invent information that is not present in the resume.
- Scores must be integers from 0 to 100.
- Keep feedback concise.
"""


SCORER_SYSTEM_PROMPT = """
You are an expert resume scoring engine.

Evaluate the resume like a professional ATS platform, recruiter,
and hiring manager.

You MUST return ONLY valid JSON.

Do NOT return:
- Markdown
- code fences
- explanations outside JSON
- headings
- commentary

The JSON MUST follow exactly this structure:

{
  "overall_score": 0,
  "summary": "",
  "scores": {
    "ats": 0,
    "content": 0,
    "impact": 0,
    "keywords": 0,
    "structure": 0,
    "readability": 0
  },
  "categories": {
    "ats": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "content": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "impact": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "keywords": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "structure": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "readability": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    }
  },
  "strengths": [],
  "weaknesses": [],
  "improvements": []
}

Each issue MUST be an object:

{
  "title": "",
  "severity": "low",
  "evidence": "",
  "recommendation": ""
}

Each improvement MUST be an object:

{
  "priority": 1,
  "title": "",
  "why": "",
  "action": ""
}

SCORING:

90-100 = excellent
75-89  = good
60-74  = warning
0-59   = poor

Important:

- Evaluate the actual resume content.
- Do not blindly score from metrics.
- Student projects, hackathons, academic work and certifications
  are legitimate evidence.
- Do not penalize a student simply for having no formal employment.
- If a job description exists, evaluate keyword alignment.
- Do not suggest irrelevant keywords.
- Be realistic.
- Scores must be integers from 0 to 100.
- Keep feedback concise.
"""


CAREER_SYSTEM_PROMPT = """
You are an AI career coach.

Analyze the candidate's resume and provide practical career guidance.

Focus on:
- current strengths
- missing skills
- suitable roles
- technical improvement areas
- project suggestions
- interview preparation
- realistic next steps

Do not invent experience.
"""


COVER_LETTER_SYSTEM_PROMPT = """
You are an expert professional cover letter writer.

Create a concise, professional, job-specific cover letter using:

- the candidate's resume
- the provided job description

Do not invent experience, achievements, companies, technologies,
metrics, or qualifications that are not present in the resume.

Keep the tone professional and natural.
"""



# ============================================================
# LLM HELPER
# ============================================================

def call_llm(
    provider: str,
    system_prompt: str,
    user_prompt: str
) -> str:

    llm = get_llm(provider)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        content = "\n".join(parts)

    return str(content)


def call_llm_with_fallback(
    preferred: str,
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Try the preferred provider first, then fall back through the
    remaining providers so one missing API key or a provider
    outage never kills a feature.
    """

    chain = [preferred] + [
        p for p in ("groq", "mistral", "openrouter")
        if p != preferred
    ]

    last_error: Optional[Exception] = None

    for provider in chain:
        try:
            return call_llm(provider, system_prompt, user_prompt)
        except Exception as e:
            last_error = e
            continue

    raise Exception(
        f"All LLM providers failed. Last error: {last_error}"
    )


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json_response(text: str) -> str:
    """
    Extract the first valid JSON object from an LLM response.

    Handles:
    - ```json ... ```
    - ``` ... ```
    - text before JSON
    - text after JSON
    """

    if not text:
        raise ValueError("LLM returned an empty response.")

    text = str(text).strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # Already JSON
    if text.startswith("{"):
        decoder = json.JSONDecoder()

        try:
            obj, _ = decoder.raw_decode(text)

            return json.dumps(obj)
        except json.JSONDecodeError:
            pass

    # Find JSON object inside surrounding text
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(
            text[start:]
        )

        return json.dumps(obj)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Invalid JSON returned by LLM: {e}"
        )


# ============================================================
# SCORE HELPERS
# ============================================================

def normalize_score(value: Any) -> int:

    try:
        value = int(float(value))
    except (ValueError, TypeError):
        return 0

    return max(
        0,
        min(
            100,
            value
        )
    )


def get_status(score: int) -> str:

    if score >= 90:
        return "excellent"

    if score >= 75:
        return "good"

    if score >= 60:
        return "warning"

    return "poor"


# ============================================================
# ISSUE NORMALIZER
# ============================================================

def normalize_issue(issue: Any) -> Dict[str, str]:

    if isinstance(issue, str):

        return {
            "title": issue,
            "severity": "medium",
            "evidence": "",
            "recommendation": ""
        }

    if not isinstance(issue, dict):

        return {
            "title": str(issue),
            "severity": "medium",
            "evidence": "",
            "recommendation": ""
        }

    severity = str(
        issue.get(
            "severity",
            "medium"
        )
    ).lower()

    if severity not in {
        "low",
        "medium",
        "high",
        "critical"
    }:
        severity = "medium"

    return {
        "title": str(
            issue.get(
                "title",
                "Resume issue"
            )
        ),
        "severity": severity,
        "evidence": str(
            issue.get(
                "evidence",
                ""
            )
        ),
        "recommendation": str(
            issue.get(
                "recommendation",
                ""
            )
        )
    }


# ============================================================
# IMPROVEMENT NORMALIZER
# ============================================================

def normalize_improvement(
    item: Any,
    index: int
) -> Dict[str, Any]:

    if isinstance(item, str):

        return {
            "priority": index,
            "title": item,
            "why": "",
            "action": item
        }

    if not isinstance(item, dict):

        return {
            "priority": index,
            "title": str(item),
            "why": "",
            "action": str(item)
        }

    try:
        priority = int(
            item.get(
                "priority",
                index
            )
        )
    except (
        ValueError,
        TypeError
    ):
        priority = index

    return {
        "priority": priority,
        "title": str(
            item.get(
                "title",
                "Resume improvement"
            )
        ),
        "why": str(
            item.get(
                "why",
                ""
            )
        ),
        "action": str(
            item.get(
                "action",
                ""
            )
        )
    }


# ============================================================
# SCORER NORMALIZATION
# ============================================================

def normalize_scorer_result(
    result: Dict[str, Any]
) -> Dict[str, Any]:

    if not isinstance(result, dict):
        raise ValueError(
            "Scorer response must be a JSON object."
        )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    raw_scores = result.get(
        "scores",
        {}
    )

    if not isinstance(
        raw_scores,
        dict
    ):
        raw_scores = {}

    scores = {
        "ats": normalize_score(
            raw_scores.get(
                "ats",
                0
            )
        ),

        "content": normalize_score(
            raw_scores.get(
                "content",
                0
            )
        ),

        "impact": normalize_score(
            raw_scores.get(
                "impact",
                0
            )
        ),

        "keywords": normalize_score(
            raw_scores.get(
                "keywords",
                0
            )
        ),

        "structure": normalize_score(
            raw_scores.get(
                "structure",
                0
            )
        ),

        "readability": normalize_score(
            raw_scores.get(
                "readability",
                0
            )
        )
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # Calculate overall score ourselves.
    #
    # This prevents the LLM from saying:
    # overall_score = 0
    # while category scores are valid.
    # --------------------------------------------------------

    overall_score = round(
        sum(scores.values()) /
        len(scores)
    )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    raw_categories = result.get(
        "categories",
        {}
    )

    if not isinstance(
        raw_categories,
        dict
    ):
        raw_categories = {}

    categories = {}

    for category_name, score in scores.items():

        raw_category = raw_categories.get(
            category_name,
            {}
        )

        if not isinstance(
            raw_category,
            dict
        ):
            raw_category = {}

        raw_issues = raw_category.get(
            "issues",
            []
        )

        if not isinstance(
            raw_issues,
            list
        ):
            raw_issues = []

        issues = [
            normalize_issue(issue)
            for issue in raw_issues
        ]

        categories[category_name] = {
            "score": score,

            "status": get_status(
                score
            ),

            "issues_count": len(
                issues
            ),

            "issues": issues
        }

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    strengths = result.get(
        "strengths",
        []
    )

    if not isinstance(
        strengths,
        list
    ):
        strengths = [str(strengths)]

    strengths = [
        str(item)
        for item in strengths
        if str(item).strip()
    ]

    # --------------------------------------------------------
    # Weaknesses
    # --------------------------------------------------------

    weaknesses = result.get(
        "weaknesses",
        []
    )

    if not isinstance(
        weaknesses,
        list
    ):
        weaknesses = [str(weaknesses)]

    weaknesses = [
        str(item)
        for item in weaknesses
        if str(item).strip()
    ]

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    raw_improvements = result.get(
        "improvements",
        []
    )

    if not isinstance(
        raw_improvements,
        list
    ):
        raw_improvements = []

    improvements = [
        normalize_improvement(
            item,
            index + 1
        )
        for index, item
        in enumerate(raw_improvements)
    ]

    # --------------------------------------------------------
    # Final guaranteed structure
    # --------------------------------------------------------

    return {
        "overall_score": overall_score,

        "summary": str(
            result.get(
                "summary",
                "Resume evaluation completed."
            )
        ),

        "scores": scores,

        "categories": categories,

        "strengths": strengths,

        "weaknesses": weaknesses,

        "improvements": improvements
    }


# ============================================================
# JSON REPAIR
# ============================================================

SCORER_JSON_REPAIR_PROMPT = """
You are a JSON repair engine.

The previous AI response was supposed to contain resume scoring
information but was not valid JSON.

Convert the supplied response into ONLY valid JSON.

Do not explain anything.

Return exactly this structure:

{
  "overall_score": 0,
  "summary": "",
  "scores": {
    "ats": 0,
    "content": 0,
    "impact": 0,
    "keywords": 0,
    "structure": 0,
    "readability": 0
  },
  "categories": {
    "ats": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "content": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "impact": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "keywords": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "structure": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    },
    "readability": {
      "score": 0,
      "status": "good",
      "issues_count": 0,
      "issues": []
    }
  },
  "strengths": [],
  "weaknesses": [],
  "improvements": []
}

Return ONLY JSON.
"""


def repair_scorer_json(
    raw_response: str
) -> Dict[str, Any]:

    repair_prompt = f"""
Here is the malformed AI response:

============================================================
MALFORMED RESPONSE
============================================================

{raw_response}

============================================================
END RESPONSE
============================================================

Convert it into the required JSON structure.
"""

    repaired = call_llm(
        "groq",
        SCORER_JSON_REPAIR_PROMPT,
        repair_prompt
    )

    cleaned = clean_json_response(
        repaired
    )

    return json.loads(
        cleaned
    )


# ============================================================
# RESUME CHECKER
# ============================================================

def resume_checker_node(
    state: ResumeState
):

    try:

        resume_text = state.get(
            "resume_text",
            ""
        )

        prompt = f"""
Analyze this resume:

============================================================
RESUME
============================================================

{resume_text}

Return ONLY the JSON structure requested by the system.
"""

        # ----------------------------------------------------
        # First Groq call
        # ----------------------------------------------------

        raw_result = call_llm(
            "groq",
            CHECKER_SYSTEM_PROMPT,
            prompt
        )

        # ----------------------------------------------------
        # First JSON parsing attempt
        # ----------------------------------------------------

        try:

            cleaned = clean_json_response(
                raw_result
            )

            parsed_result = json.loads(
                cleaned
            )

        except Exception:

            # ------------------------------------------------
            # Second Groq call to repair response
            # ------------------------------------------------

            parsed_result = repair_scorer_json(
                raw_result
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not isinstance(
            parsed_result,
            dict
        ):
            raise ValueError(
                "Checker did not return a JSON object."
            )

        # ----------------------------------------------------
        # Normalize (same contract as the scorer)
        # ----------------------------------------------------

        analysis = normalize_scorer_result(
            parsed_result
        )

        return {
            "result": {
                "feature": "checker",
                "analysis": analysis
            }
        }

    except Exception as e:

        return {
            "error": (
                "Resume checker failed: "
                + str(e)
            )
        }


# ============================================================
# RESUME SCORER
# ============================================================

def resume_scorer_node(
    state: ResumeState
):

    try:

        resume_text = state.get(
            "resume_text",
            ""
        )

        job_description = state.get(
            "job_description",
            ""
        )

        resume_data = state.get(
            "resume_data",
            {}
        )

        metrics = resume_data.get(
            "metrics",
            {}
        )

        sections = resume_data.get(
            "sections",
            {}
        )

        contact = resume_data.get(
            "contact",
            {}
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = f"""
Evaluate this resume as a professional resume scoring platform.

============================================================
RESUME
============================================================

{resume_text}

============================================================
JOB DESCRIPTION
============================================================

{
    job_description.strip()
    if job_description.strip()
    else
    "No job description provided."
}

============================================================
RESUME METRICS
============================================================

Word count:
{metrics.get("word_count", "unknown")}

Line count:
{metrics.get("line_count", "unknown")}

Bullet count:
{metrics.get("bullet_count", "unknown")}

Quantified bullet count:
{metrics.get("quantified_bullet_count", "unknown")}

Quantified bullet percentage:
{metrics.get("quantified_bullet_percentage", "unknown")}

============================================================
DETECTED SECTIONS
============================================================

{list(sections.keys())}

============================================================
CONTACT INFORMATION
============================================================

{contact}

============================================================
EVALUATION RULES
============================================================

Evaluate the actual resume.

Use metrics as supporting evidence.

Do not score only from metrics.

For student resumes:
- projects count as valid experience
- hackathons count as valid evidence
- academic work counts as valid evidence
- certifications count as valid evidence
- do not penalize absence of formal employment

If a job description is provided:
- evaluate keyword alignment
- identify meaningful keyword gaps
- do not recommend irrelevant keywords

Return ONLY the JSON structure requested by the system.
"""

        # ----------------------------------------------------
        # First Groq call
        # ----------------------------------------------------

        raw_result = call_llm(
            "groq",
            SCORER_SYSTEM_PROMPT,
            prompt
        )

        # ----------------------------------------------------
        # First JSON parsing attempt
        # ----------------------------------------------------

        try:

            cleaned = clean_json_response(
                raw_result
            )

            parsed_result = json.loads(
                cleaned
            )

        except Exception:

            # ------------------------------------------------
            # Second Groq call to repair response
            # ------------------------------------------------

            parsed_result = repair_scorer_json(
                raw_result
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not isinstance(
            parsed_result,
            dict
        ):
            raise ValueError(
                "Scorer did not return a JSON object."
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        analysis = normalize_scorer_result(
            parsed_result
        )

        # ----------------------------------------------------
        # CRITICAL API CONTRACT
        # ----------------------------------------------------

        return {
            "result": {
                "feature": "scorer",
                "analysis": analysis
            }
        }

    except Exception as e:

        return {
            "error": (
                "Resume scorer failed: "
                + str(e)
            )
        }


# ============================================================
# CAREER COACH
# ============================================================

def career_coach_node(
    state: ResumeState
):

    try:

        resume_text = state.get(
            "resume_text",
            ""
        )

        job_description = state.get(
            "job_description",
            ""
        )

        prompt = f"""
Analyze the following resume and provide career guidance.

============================================================
RESUME
============================================================

{resume_text}

============================================================
ADDITIONAL CONTEXT (target role / candidate's question)
============================================================

{
    job_description.strip()
    if job_description.strip()
    else
    "No additional context provided."
}

If a candidate question is included above, answer it directly
and specifically, using the resume as evidence.
"""

        raw_result = call_llm_with_fallback(
            "openrouter",
            CAREER_SYSTEM_PROMPT,
            prompt
        )

        return {
            "result": {
                "feature": "career_coach",
                "analysis": raw_result
            }
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# COVER LETTER
# ============================================================

def cover_letter_node(
    state: ResumeState
):

    try:

        resume_text = state.get(
            "resume_text",
            ""
        )

        job_description = state.get(
            "job_description",
            ""
        )

        prompt = f"""
RESUME
============================================================

{resume_text}

============================================================
JOB DESCRIPTION
============================================================

{job_description}
"""

        raw_result = call_llm_with_fallback(
            "mistral",
            COVER_LETTER_SYSTEM_PROMPT,
            prompt
        )

        return {
            "result": {
                "feature": "cover_letter",
                "analysis": raw_result
            }
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# ROUTER
# ============================================================

def feature_router(
    state: ResumeState
):

    feature = state.get(
        "feature",
        ""
    )

    if feature == "checker":
        return "checker"

    if feature == "scorer":
        return "scorer"

    if feature == "career_coach":
        return "career_coach"

    if feature == "cover_letter":
        return "cover_letter"

    return "scorer"


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(
    ResumeState
)

workflow.add_node(
    "checker",
    resume_checker_node
)

workflow.add_node(
    "scorer",
    resume_scorer_node
)

workflow.add_node(
    "career_coach",
    career_coach_node
)

workflow.add_node(
    "cover_letter",
    cover_letter_node
)

workflow.set_conditional_entry_point(
    feature_router,
    {
        "checker": "checker",
        "scorer": "scorer",
        "career_coach": "career_coach",
        "cover_letter": "cover_letter"
    }
)

workflow.add_edge(
    "checker",
    END
)

workflow.add_edge(
    "scorer",
    END
)

workflow.add_edge(
    "career_coach",
    END
)

workflow.add_edge(
    "cover_letter",
    END
)

resume_graph = workflow.compile()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_resume_feature(
    feature: str,
    resume_text: str,
    job_description: str = "",
    resume_data: Optional[
        Dict[str, Any]
    ] = None
):

    state: ResumeState = {
        "feature": feature,
        "resume_text": resume_text,
        "job_description": job_description,
        "resume_data": resume_data or {}
    }

    result = resume_graph.invoke(
        state
    )

    if result.get(
        "error"
    ):

        raise Exception(
            result["error"]
        )

    return result.get(
        "result"
    )