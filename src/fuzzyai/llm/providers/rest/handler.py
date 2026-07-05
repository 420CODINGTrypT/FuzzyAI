"""REST provider with SSRF protection and secure defaults."""

import ipaddress
import json
import logging
import re
import socket
from typing import Any, AsyncGenerator
from urllib.parse import urlparse

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Blocked networks for SSRF prevention
BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

BLOCKED_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
BLOCKED_TLDS = {".local", ".internal", ".corp", ".home", ".lan"}


def validate_url(url: str) -> bool:
    """Validate URL to prevent SSRF attacks."""
    try:
        parsed = urlparse(url)
        
        # Only allow http/https
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"Blocked non-HTTP(S) protocol: {parsed.scheme}")
            return False
        
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # Block known internal hostnames
        if hostname.lower() in BLOCKED_HOSTS:
            logger.warning(f"Blocked internal hostname: {hostname}")
            return False
        
        # Block internal TLDs
        hostname_lower = hostname.lower()
        for tld in BLOCKED_TLDS:
            if hostname_lower.endswith(tld):
                logger.warning(f"Blocked internal TLD: {tld}")
                return False
        
        # Block URLs with embedded credentials
        if parsed.username or parsed.password:
            logger.warning("Blocked URL with embedded credentials")
            return False
        
        # Block internal IP ranges
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            for network in BLOCKED_NETWORKS:
                if ip in network:
                    logger.warning(f"Blocked internal IP: {ip}")
                    return False
        except (socket.gaierror, ValueError):
            # If we can't resolve, allow it (could be external hostname)
            pass
        
        return True
    except Exception as e:
        logger.warning(f"URL validation error: {e}")
        return False


def _resolve_path_safely(file_path: str) -> str:
    """Resolve and validate a file path to prevent path traversal."""
    from pathlib import Path
    
    p = Path(file_path).resolve()
    cwd = Path.cwd().resolve()
    
    # Ensure the path is within the current working directory or project
    try:
        p.relative_to(cwd)
    except ValueError:
        # Try project root
        project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
        try:
            p.relative_to(project_root)
        except ValueError:
            raise ValueError(f"Path traversal detected: {file_path} is outside project directory")
    
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    
    return str(p)


class RestProvider(BaseLLMProvider):
    """REST LLM provider with SSRF protection."""
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._host = extra.get("host", "localhost")
        self._port = extra.get("port", 8000)
        self._scheme = extra.get("scheme", "http")
        self._path = extra.get("path", "/")
        
        # Validate the constructed URL
        self._url = f"{self._scheme}://{self._host}:{self._port}{self._path}"
        if not validate_url(self._url):
            raise ValueError(f"URL failed SSRF validation: {self._url}")
        
        # If model is a file, validate it safely
        if model and not model.startswith(("http://", "https://")):
            self._http_file = _resolve_path_safely(model)
        else:
            self._http_file = None
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add any API key from headers extra param
        if "headers" in extra and isinstance(extra["headers"], dict):
            headers.update(extra["headers"])
        
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **{k: v for k, v in extra.items() if k not in ["host", "port", "scheme", "path", "headers"]}
        }
        
        try:
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
                async with session.post(self._url, headers=headers, json=body) as response:
                    if response.status != 200:
                        logger.error(f"REST API error: {response.status}")
                        raise RuntimeError(f"REST API error: {response.status}")
                    
                    data = await response.json()
                    
                    # Try to extract response from common formats
                    if "choices" in data:
                        for choice in data["choices"]:
                            content = choice.get("message", {}).get("content", "")
                            yield type('Response', (), {'response': content})()
                    elif "response" in data:
                        yield type('Response', (), {'response': data["response"]})()
                    elif "text" in data:
                        yield type('Response', (), {'response': data["text"]})()
                    elif "content" in data:
                        yield type('Response', (), {'response': data["content"]})()
                    else:
                        yield type('Response', (), {'response': str(data)})()
        except aiohttp.ClientError as e:
            logger.error(f"REST provider request failed: {type(e).__name__}")
            raise RuntimeError(f"REST provider request failed") from e
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> str:
        return "Custom REST API (configure via host, port, scheme, path)"
    
    @classmethod
    def get_supported_models(cls) -> str:
        return cls.supported_models()


register_provider(LLMProvider.REST, RestProvider)
