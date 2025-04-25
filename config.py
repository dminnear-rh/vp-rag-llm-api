import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer


class ModelType(str, Enum):
    """
    Enumeration of supported model backends.

    Attributes:
        VLLM: A local or hosted VLLM-compatible model.
        OPENAI: An OpenAI model (e.g., gpt-4o, gpt-4o-mini).
    """

    VLLM = "vllm"
    OPENAI = "openai"


@dataclass
class ModelConfig:
    """
    Configuration for a specific LLM model.

    Attributes:
        name: Unique model identifier (used in model selection).
        url: Base URL for querying the model (if self-hosted).
        model_type: Type of model (e.g., VLLM, OPENAI).
        max_total_tokens: Maximum number of tokens allowed for this model (input + output).
    """

    name: str
    url: str
    model_type: ModelType
    max_total_tokens: int


@dataclass
class AppConfig:
    """
    Application configuration loaded from environment variables.

    Attributes:
        qdrant_url: URL to the Qdrant instance.
        qdrant_collection: Name of the Qdrant collection to search.
        models: Dictionary of available models by name.
        default_model: Name of the default model to use.
        log_level: Logging level (e.g., logging.INFO).
        openai_api_key: API key for OpenAI access (if applicable).
        embedder: SentenceTransformer instance for embedding queries/documents.
        cross_encoder: CrossEncoder instance for re-ranking vector search results.
        system_message: System prompt injected into every chat session.
        max_response_tokens: Token budget for LLM-generated answers.
    """

    qdrant_url: str
    qdrant_collection: str
    models: Dict[str, ModelConfig]
    default_model: str
    log_level: int
    openai_api_key: str
    embedder: SentenceTransformer
    cross_encoder: CrossEncoder
    system_message: str
    max_response_tokens: int

    @staticmethod
    def _get_required_env_var(key: str) -> str:
        """
        Retrieve a required environment variable or raise an error.

        Args:
            key: Environment variable name.

        Returns:
            The value of the environment variable.

        Raises:
            ValueError: If the variable is missing or empty.
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @staticmethod
    def _parse_log_level(log_level_name: str) -> int:
        """
        Convert log level string to logging constant.

        Args:
            log_level_name: Log level name (e.g., "debug", "info").

        Returns:
            Corresponding `logging` module level.

        Raises:
            ValueError: If the log level name is invalid.
        """
        log_levels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        if log_level_name not in log_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: '{log_level_name}'. Must be one of: {', '.join(log_levels.keys())}"
            )
        return log_levels[log_level_name]

    @staticmethod
    def _parse_model_registry(raw: str) -> Dict[str, ModelConfig]:
        """
        Parse a JSON string of model configurations into a dictionary.

        Args:
            raw: JSON string from `VLLM_MODELS` environment variable.

        Returns:
            Dictionary mapping model names to `ModelConfig` objects.

        Raises:
            ValueError: If the input JSON is malformed or missing required fields.
        """
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid VLLM_MODELS JSON: {e}")

        models = {}
        for model in parsed:
            try:
                config = ModelConfig(
                    name=model["name"],
                    url=model["url"],
                    model_type=ModelType(model["model_type"]),
                    max_total_tokens=int(model["max_total_tokens"]),
                )
                models[config.name] = config
            except KeyError as e:
                raise ValueError(f"Missing required field in model definition: {e}")
        return models

    @staticmethod
    def load() -> "AppConfig":
        """
        Load application configuration from environment variables.

        Expected environment variables:
            - QDRANT_URL
            - QDRANT_COLLECTION
            - DEFAULT_MODEL
            - VLLM_MODELS (JSON string list)
            - OPENAI_API_KEY
            - EMBEDDING_MODEL
            - CROSS_ENCODER_MODEL
            - SYSTEM_MESSAGE
            - MAX_RESPONSE_TOKENS
            - LOG_LEVEL

        Returns:
            Fully initialized `AppConfig` object.

        Raises:
            ValueError: If required environment variables are missing or invalid.
        """
        load_dotenv()
        get = AppConfig._get_required_env_var

        log_level = AppConfig._parse_log_level(get("LOG_LEVEL").lower())
        logging.basicConfig(level=log_level)
        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized.")

        qdrant_url = get("QDRANT_URL")
        qdrant_collection = get("QDRANT_COLLECTION")
        default_model = get("DEFAULT_MODEL")
        models = AppConfig._parse_model_registry(get("VLLM_MODELS"))
        if default_model not in models:
            raise ValueError(
                f"DEFAULT_MODEL '{default_model}' not found in VLLM_MODELS list"
            )

        openai_api_key = get("OPENAI_API_KEY")

        embedding_model = SentenceTransformer(get("EMBEDDING_MODEL"))
        cross_encoder = CrossEncoder(get("CROSS_ENCODER_MODEL"))

        system_message = get("SYSTEM_MESSAGE")
        max_response_tokens = int(get("MAX_RESPONSE_TOKENS"))

        return AppConfig(
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            models=models,
            default_model=default_model,
            log_level=log_level,
            openai_api_key=openai_api_key,
            embedder=embedding_model,
            cross_encoder=cross_encoder,
            system_message=system_message,
            max_response_tokens=max_response_tokens,
        )
