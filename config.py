import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from dotenv import load_dotenv


class ModelType(str, Enum):
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelConfig:
    name: str
    url: str
    api_key: Optional[str]
    model_type: ModelType
    max_total_tokens: int


@dataclass
class AppConfig:
    qdrant_url: str
    qdrant_collection: str
    models: Dict[str, ModelConfig]
    default_model: str
    log_level: int

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
                    api_key=model.get("api_key"),
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

        log_level = AppConfig._parse_log_level(os.getenv("LOG_LEVEL", "info").lower())
        logging.basicConfig(level=log_level)
        logger = logging.getLogger(__name__)
        logger.debug("Logging initialized.")

        qdrant_url = AppConfig._get_required_env_var("QDRANT_URL")
        qdrant_collection = AppConfig._get_required_env_var("QDRANT_COLLECTION")
        models = AppConfig._parse_model_registry(
            AppConfig._get_required_env_var("VLLM_MODELS")
        )
        default_model = AppConfig._get_required_env_var("DEFAULT_MODEL")

        if default_model not in models:
            raise ValueError(
                f"DEFAULT_MODEL '{default_model}' not found in VLLM_MODELS list"
            )

        return AppConfig(
            qdrant_url=qdrant_url,
            qdrant_collection=qdrant_collection,
            models=models,
            default_model=default_model,
            log_level=log_level,
        )
