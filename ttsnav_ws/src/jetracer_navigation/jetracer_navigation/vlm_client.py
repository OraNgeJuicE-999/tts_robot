#!/usr/bin/env python3
"""Importable session-aware client for backend command and image VLM APIs.

This client connects to the FastAPI backend, not directly to the VLM/vLLM
model port.

Install on the robot:

    python -m pip install -r requirements-clients.txt

Request a robot command:

    python scripts/vlm_client.py \
        --base-url https://140.96.96.16:8443 \
        --ca-file certs/dev-ca.pem \
        command "Go to the top-left corner and inspect for oil leaks"

Upload an image and wait for its analysis:

    python scripts/vlm_client.py \
        --base-url https://140.96.96.16:8443 \
        --ca-file certs/dev-ca.pem \
        image ./frame.jpg \
        --prompt "Inspect the floor for nails and fallen objects"

Use plain HTTP on a trusted local network:

    python scripts/vlm_client.py \
        --base-url http://SERVER_IP:8080 \
        command "Stop now"

Use from another Python application:

    async with VLMClient(
        "https://140.96.96.16:8443",
        ca_file="certs/dev-ca.pem",
    ) as client:
        session = await client.start_session()
        command = await client.get_robot_command("Go to the top left")
        analysis = await client.analyze_image(
            "frame.jpg",
            "Inspect the floor for nails",
        )

Run ``python scripts/vlm_client.py --help`` for every option. Global options
must appear before the ``command`` or ``image`` operation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import ssl
import sys
from typing import Any
from urllib.parse import quote, urlparse

import httpx


DEFAULT_BASE_URL = "https://140.96.96.16:8443"
MAX_IMAGE_SIZE = 10 * 1024 * 1024
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

CLI_EXAMPLES = """examples:
  Request a command:
    python scripts/vlm_client.py --base-url http://SERVER_IP:8080 command "Go top left"

  Analyze an image:
    python scripts/vlm_client.py --base-url http://SERVER_IP:8080 image frame.jpg \\
      --prompt "Inspect the floor for nails"

  Use HTTPS with the development CA:
    python scripts/vlm_client.py --base-url https://SERVER_IP:8443 \\
      --ca-file certs/dev-ca.pem command "Stop now"

  Resume an existing inspection session:
    python scripts/vlm_client.py --base-url http://SERVER_IP:8080 \\
      --session-id sess_123 command "Inspect for oil leaks"
"""


class VLMClientError(RuntimeError):
    """The backend VLM workflow could not complete normally."""


class VLMClient:
    """Own one inspection session and expose command/image JSON methods."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        ca_file: str | Path | None = None,
        insecure: bool = False,
        dog_id: str = "AGX-Orin-Dog-01",
        location: str = "Factory-Floor-A",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        analysis_poll_seconds: float = 0.5,
        analysis_timeout: float = 120,
    ) -> None:
        if analysis_poll_seconds <= 0 or analysis_timeout <= 0:
            raise ValueError("analysis polling and timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.dog_id = dog_id
        self.location = location
        self.metadata = metadata or {}
        self.session_id = session_id
        self.analysis_poll_seconds = analysis_poll_seconds
        self.analysis_timeout = analysis_timeout
        verify = self._http_verify(ca_file, insecure)
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            verify=verify,
            timeout=httpx.Timeout(30, connect=10),
        )

    async def __aenter__(self) -> "VLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _http_verify(
        self, ca_file: str | Path | None, insecure: bool
    ) -> bool | str:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.scheme == "http":
            if ca_file or insecure:
                raise ValueError(
                    "ca_file and insecure apply only to HTTPS URLs"
                )
            return True
        if insecure:
            return False
        selected = Path(ca_file) if ca_file is not None else None
        repository_ca = (
            Path(__file__).resolve().parents[1] / "certs" / "dev-ca.pem"
        )
        if selected is None and repository_ca.is_file():
            selected = repository_ca
        if selected is not None and not selected.is_file():
            raise ValueError(f"CA file does not exist: {selected}")
        if selected is not None:
            # Validate the bundle eagerly so configuration errors fail locally.
            ssl.create_default_context(cafile=str(selected))
            return str(selected)
        return True

    async def start_session(self) -> dict[str, Any]:
        """Create a session, or validate and resume the configured session."""
        if self.session_id:
            response = await self._http.get(
                f"/api/v1/sessions/{quote(self.session_id, safe='')}"
            )
            if response.status_code == 200:
                return self._json(response)
            if response.status_code != 404:
                self._raise(response)
            self.session_id = None

        response = await self._http.post(
            "/api/v1/sessions",
            json={
                "dog_id": self.dog_id,
                "location": self.location,
                "metadata": self.metadata,
            },
        )
        self._raise(response)
        result = self._json(response)
        self.session_id = str(result["session_id"])
        return result

    async def close(self) -> None:
        await self._http.aclose()

    async def get_robot_command(
        self, transcript: str
    ) -> dict[str, Any]:
        """Convert one final ASR transcript into validated robot-command JSON."""
        normalized = transcript.strip()
        if not normalized:
            raise ValueError("transcript must not be blank")
        if len(normalized) > 4000:
            raise ValueError("transcript exceeds 4000 characters")
        session_id = await self._require_session()
        response = await self._http.post(
            f"/api/v1/sessions/{quote(session_id, safe='')}/commands",
            json={"transcript": normalized},
        )
        self._raise(response)
        return self._json(response)

    async def analyze_image(
        self,
        image: str | Path | bytes,
        prompt: str,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload one prompted image and return terminal analysis JSON."""
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("prompt must not be blank")
        if len(normalized_prompt) > 2000:
            raise ValueError("prompt exceeds 2000 characters")

        content, selected_filename, selected_mime = self._prepare_image(
            image, filename=filename, mime_type=mime_type
        )
        session_id = await self._require_session()
        digest = hashlib.sha256(
            content + b"\0" + normalized_prompt.encode("utf-8")
        ).hexdigest()
        response = await self._http.post(
            f"/api/v1/sessions/{quote(session_id, safe='')}/images",
            files={
                "file": (selected_filename, content, selected_mime),
            },
            data={"prompt": normalized_prompt},
            headers={"Idempotency-Key": f"vlm-{digest}"},
        )
        self._raise(response)
        image_id = str(self._json(response)["image_id"])

        deadline = (
            asyncio.get_running_loop().time() + self.analysis_timeout
        )
        while True:
            response = await self._http.get(
                f"/api/v1/images/{quote(image_id, safe='')}/analysis"
            )
            self._raise(response)
            result = self._json(response)
            if result.get("status") in {"COMPLETED", "FAILED"}:
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise VLMClientError(
                    f"analysis timed out for image {image_id}"
                )
            await asyncio.sleep(self.analysis_poll_seconds)

    async def _require_session(self) -> str:
        if not self.session_id:
            await self.start_session()
        assert self.session_id is not None
        return self.session_id

    @staticmethod
    def _prepare_image(
        image: str | Path | bytes,
        *,
        filename: str | None,
        mime_type: str | None,
    ) -> tuple[bytes, str, str]:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise ValueError(f"image file does not exist: {path}")
            content = path.read_bytes()
            selected_filename = filename or path.name
            guessed_mime = mimetypes.guess_type(path.name)[0]
        elif isinstance(image, bytes):
            content = image
            selected_filename = filename or "capture.jpg"
            guessed_mime = mimetypes.guess_type(selected_filename)[0]
        else:
            raise TypeError("image must be a path or bytes")

        if not content:
            raise ValueError("image must not be empty")
        if len(content) > MAX_IMAGE_SIZE:
            raise ValueError("image exceeds 10 MB")
        detected_mime = VLMClient._detect_mime(content)
        selected_mime = mime_type or guessed_mime or detected_mime
        if (
            selected_mime not in SUPPORTED_MIME_TYPES
            or selected_mime != detected_mime
        ):
            raise ValueError(
                "image bytes and MIME type must be JPEG, PNG, or WebP"
            )
        return content, selected_filename, selected_mime

    @staticmethod
    def _detect_mime(content: bytes) -> str | None:
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if (
            len(content) >= 12
            and content[:4] == b"RIFF"
            and content[8:12] == b"WEBP"
        ):
            return "image/webp"
        return None

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            result = response.json()
        except ValueError as exc:
            raise VLMClientError("backend returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise VLMClientError("backend returned non-object JSON")
        return result

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise VLMClientError(
            f"backend request failed ({response.status_code}): {detail}"
        )


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("metadata must be valid JSON") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("metadata must be a JSON object")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send command transcripts and prompted images to the FastAPI "
            "backend."
        ),
        epilog=CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DOG_SERVER_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--ca-file", type=Path)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--dog-id", default="AGX-Orin-Dog-01")
    parser.add_argument("--location", default="Factory-Floor-A")
    parser.add_argument("--metadata", type=parse_json_object, default={})
    parser.add_argument("--session-id")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    command = subparsers.add_parser("command")
    command.add_argument("transcript")
    image = subparsers.add_parser("image")
    image.add_argument("path", type=Path)
    image.add_argument("--prompt", required=True)
    return parser.parse_args()


async def run_main(args: argparse.Namespace) -> None:
    async with VLMClient(
        args.base_url,
        ca_file=args.ca_file,
        insecure=args.insecure,
        dog_id=args.dog_id,
        location=args.location,
        metadata=args.metadata,
        session_id=args.session_id,
    ) as client:
        await client.start_session()
        if args.operation == "command":
            result = await client.get_robot_command(args.transcript)
        else:
            result = await client.analyze_image(
                args.path, args.prompt
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> int:
    try:
        asyncio.run(run_main(parse_args()))
    except (OSError, ValueError, VLMClientError) as exc:
        print(f"VLM client error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
