"""
_lib/auth.py
------------
Google ADC authentication and OpenAI-compatible client factory.
Credentials are loaded once at import time; the token is refreshed on demand.
"""

import google.auth
import google.auth.transport.requests
from openai import OpenAI

from . import config

# Load Application Default Credentials once for the whole process
_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_auth_request = google.auth.transport.requests.Request()


def get_client(model_cfg: dict) -> OpenAI:
    """Return a fresh OpenAI client with a valid bearer token for *model_cfg*."""
    if not _credentials.valid:
        _credentials.refresh(_auth_request)
    base_url = (
        f"https://{model_cfg['endpoint']}/v1/projects/{config.PROJECT_ID}"
        f"/locations/{model_cfg['region']}/endpoints/openapi"
    )
    return OpenAI(base_url=base_url, api_key=_credentials.token)
