import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer


class ModelType(str, Enum):
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    name: str
    url: str
    model_type: ModelType
    max_total_tokens: int


@dataclass
class AppConfig:
    qdrant_url: str
    qdrant_collection: str
    models: Dict[str, ModelConfig]
    default_model: str
    log_level: int
    openai_api_key: str
    embedder: SentenceTransformer
    cross_encoder: CrossEncoder

    @staticmethod
    def _get_required_env_var(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @staticmethod
    def _parse_log_level(log_level_name: str) -> int:
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

        return AppConfig(
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            models=models,
            default_model=default_model,
            log_level=log_level,
            openai_api_key=openai_api_key,
            embedder=embedding_model,
            cross_encoder=cross_encoder,
        )
