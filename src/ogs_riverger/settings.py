from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Environment settings.
    EFAS_FTP_USER: Username to the EFAS (European Flood Awareness System)
        service.
    EFAS_FTP_PASSWORD: Password to the EFAS service.

    CDS_API_URL: Copernicus CDS (Climate Data Store) API URL.
    CDS_API_KEY: Copernicus CDS API key.
    """

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    CDS_API_URL: str | None = None
    CDS_API_KEY: SecretStr | None = None

    EFAS_FTP_USER: str | None = None
    EFAS_FTP_PASSWORD: SecretStr | None = None
