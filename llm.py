import os
from typing import Optional

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI


# Load environment variables from .env
load_dotenv()


# ============================================================
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# You can change these models from one place.
# Keep model names in environment variables if you want
# to change them without modifying the code.

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-large-latest"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)


# ============================================================
# VALIDATE API KEYS
# ============================================================

def _check_api_key(key: Optional[str], provider: str) -> None:
    """
    Check whether an API key exists.

    This is intentionally called when a provider is requested,
    rather than when this module is imported, so that one missing
    provider key does not prevent the whole application from starting.
    """

    if not key:
        raise ValueError(
            f"{provider} API key is missing. "
            f"Please add it to your .env file."
        )


# ============================================================
# GROQ
# ============================================================

def get_groq_llm(
    temperature: float = 0.2,
    model: Optional[str] = None
):
    """
    Return a Groq chat model.

    Groq is useful when you want fast responses,
    especially for Resume Checker and Career Coach.
    """

    _check_api_key(GROQ_API_KEY, "Groq")

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model or GROQ_MODEL,
        temperature=temperature,
    )


# ============================================================
# MISTRAL
# ============================================================

def get_mistral_llm(
    temperature: float = 0.2,
    model: Optional[str] = None
):
    """
    Return a Mistral chat model.
    """

    _check_api_key(MISTRAL_API_KEY, "Mistral")

    return ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=model or MISTRAL_MODEL,
        temperature=temperature,
    )


# ============================================================
# OPENROUTER
# ============================================================

def get_openrouter_llm(
    temperature: float = 0.2,
    model: Optional[str] = None
):
    """
    Return an OpenRouter chat model.

    OpenRouter uses an OpenAI-compatible API,
    therefore ChatOpenAI can be configured with
    OpenRouter's base URL.
    """

    _check_api_key(OPENROUTER_API_KEY, "OpenRouter")

    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=model or OPENROUTER_MODEL,
        temperature=temperature,
    )


# ============================================================
# PROVIDER SELECTOR
# ============================================================

def get_llm(
    provider: str = "groq",
    temperature: float = 0.2,
    model: Optional[str] = None
):
    """
    Get an LLM based on the provider name.

    Example:

        llm = get_llm("groq")

        llm = get_llm("mistral")

        llm = get_llm("openrouter")
    """

    provider = provider.lower().strip()

    if provider == "groq":
        return get_groq_llm(
            temperature=temperature,
            model=model
        )

    if provider == "mistral":
        return get_mistral_llm(
            temperature=temperature,
            model=model
        )

    if provider in {"openrouter", "open-router"}:
        return get_openrouter_llm(
            temperature=temperature,
            model=model
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        f"Choose from: groq, mistral, openrouter."
    )


# ============================================================
# SIMPLE TEXT GENERATION
# ============================================================

def generate_text(
    prompt: str,
    provider: str = "groq",
    temperature: float = 0.2,
    model: Optional[str] = None
) -> str:
    """
    Send a simple prompt to the selected LLM
    and return the generated text.

    This is useful for simple tasks where you don't
    need a LangGraph state or multiple messages.
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    llm = get_llm(
        provider=provider,
        temperature=temperature,
        model=model
    )

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# MESSAGE GENERATION
# ============================================================

def generate_from_messages(
    messages: list[BaseMessage],
    provider: str = "groq",
    temperature: float = 0.2,
    model: Optional[str] = None
) -> str:
    """
    Generate a response from LangChain messages.

    Useful when your LangGraph nodes already work with
    SystemMessage / HumanMessage / AIMessage.
    """

    if not messages:
        raise ValueError("Messages cannot be empty.")

    llm = get_llm(
        provider=provider,
        temperature=temperature,
        model=model
    )

    response = llm.invoke(messages)

    return response.content


# ============================================================
# PROVIDER HEALTH CHECK
# ============================================================

def check_provider(provider: str) -> bool:
    """
    Check whether the requested provider has an API key configured.

    This does not make an API request.
    It only checks configuration.
    """

    provider = provider.lower().strip()

    if provider == "groq":
        return bool(GROQ_API_KEY)

    if provider == "mistral":
        return bool(MISTRAL_API_KEY)

    if provider in {"openrouter", "open-router"}:
        return bool(OPENROUTER_API_KEY)

    return False


# ============================================================
# AVAILABLE PROVIDERS
# ============================================================

def get_available_providers() -> list[str]:
    """
    Return providers whose API keys are configured.
    """

    providers = []

    if GROQ_API_KEY:
        providers.append("groq")

    if MISTRAL_API_KEY:
        providers.append("mistral")

    if OPENROUTER_API_KEY:
        providers.append("openrouter")

    return providers