"""---------------------------------------------------------------

Config management for llm-api

-----------------------------------------------------------------"""
#%%

import logging
import os
from typing import Any, Dict, Optional, Tuple, Type

import yaml
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic.fields import FieldInfo

logger = logging.getLogger("llm-api.config")

#%%

class Settings(BaseSettings):  # pylint: disable=too-few-public-methods
    """
    Settings class to configure the app
    """

    model_dir: str = "./models"
    model_family: str 
    model_params: Optional[Dict[str, Any]] = {}
    setup_params: Dict[str, Any] = {}
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    #model_config = SettingsConfigDict(env_prefix="LLM_API_")

    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=".env", env_file_encoding="utf-8", extra='ignore'
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_config_settings_source(settings_cls),
        )

#%%

class yaml_config_settings_source(PydanticBaseSettingsSource):
    """
    YAML file settings source
    """

    def __init__(self, settings_cls: Type[BaseSettings]):
        super().__init__(settings_cls)

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> Tuple[Any, str, bool]:
        """
        Get the value of a field.
        """

        return field, field_name, False

    def __call__(self) -> Dict[str, Any]:
        """
        
        """
        if not os.path.exists("config.yaml"):
            logger.warning("no config file found")
            return {}
        try:
            logger.info("loading config file config.yaml")
            with open("config.yaml", encoding="utf-8") as conf_file:
                data = yaml.load(conf_file, Loader=yaml.FullLoader)
            if data is None:
                logger.warning("config file is empty")
                return {}
            logger.info(str(data))
            return data
        except yaml.YAMLError as exp:
            raise exp


#%%

settings = Settings()

# %%
