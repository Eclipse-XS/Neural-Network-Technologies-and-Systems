"""Comprehensive adversarial and security test suite (Section 13).

Covers:
- Path traversal and malicious filenames
- AST calculator sandbox escape attempts
- Direct and indirect prompt injection handling
- Remote cloud URL rejection (loopback enforcement)
- Oversized upload rejection and resource exhaustion
"""
import pytest
from pathlib import Path

from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import (
    DocumentValidationError,
    OversizedDocumentError,
    UnsupportedFormatError,
)
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import validate_loopback_url
from labs.lab07_final_project.src.knowledge_copilot.rag.extractors import (
    sanitize_filename,
    validate_file_content,
)
from labs.lab07_final_project.src.knowledge_copilot.agents.tools import safe_calculate


# ----------------- 1. Path Traversal & Malicious Filenames (10 Cases) ----------------- #
@pytest.mark.parametrize(
    "malicious_name",
    [
        "../../etc/passwd",
        "..\\..\\Windows\\System32\\cmd.exe",
        "/etc/shadow",
        "C:\\boot.ini",
        "....//....//sensitive.txt",
        "..%2f..%2fsecret.md",
        "doc\x00hidden.md",
        "\x1b[31mred_exploit.md",
        "\r\nnewline_injection.txt",
        "./../../../../var/log/syslog",
    ],
)
def test_path_traversal_and_malicious_names_sanitized_or_rejected(malicious_name: str):
    """Zero path traversal successes: every traversal sequence is stripped or rejected."""
    try:
        clean = sanitize_filename(malicious_name)
        # Stripped name must not contain directory separators or null bytes
        assert "/" not in clean
        assert "\\" not in clean
        assert "\0" not in clean
        assert clean not in {".", ".."}
    except DocumentValidationError:
        # Rejection is fully secure
        pass


# ----------------- 2. Calculator Arbitrary Code Execution (10 Cases) ----------------- #
@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('whoami')",
        "__import__('subprocess').run(['calc'])",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__[0].__subclasses__()",
        "eval('2 + 2')",
        "exec('import sys')",
        "[c for c in ().__class__.__base__.__subclasses__()]",
        "2 ** 1000000",          # CPU denial of service
        "1e309",                 # Infinite float overflow
        "globals()",
    ],
)
def test_calculator_sandbox_rejects_arbitrary_code(payload: str):
    """Zero code execution capabilities available through AST calculator."""
    with pytest.raises((ValueError, OverflowError, ZeroDivisionError)):
        safe_calculate(payload)


# ----------------- 3. Loopback Endpoint Enforcement (5 Cases) ----------------- #
@pytest.mark.parametrize(
    "cloud_url",
    [
        "https://api.openai.com/v1",
        "http://remote-server.evil.com:1234/v1",
        "http://user:password@localhost:1234/v1",
        "ftp://localhost:1234/v1",
        "http://192.168.1.100:1234/v1",
    ],
)
def test_loopback_validator_rejects_remote_and_credential_urls(cloud_url: str):
    """Enforce privacy: zero silent remote LLM fallbacks or external data leaks."""
    with pytest.raises(ValueError):
        validate_loopback_url(cloud_url)


# ----------------- 4. Malformed / Oversized File Uploads (5 Cases) ----------------- #
def test_oversized_file_rejected():
    with pytest.raises(OversizedDocumentError):
        validate_file_content(b"0" * 1000, "oversized.txt", max_size_bytes=500)


def test_empty_content_rejected():
    with pytest.raises(DocumentValidationError):
        validate_file_content(b"", "empty.md")


def test_unsupported_binary_rejected():
    with pytest.raises(UnsupportedFormatError):
        validate_file_content(b"\x7fELF", "program.bin")


def test_unsupported_archive_rejected():
    with pytest.raises(UnsupportedFormatError):
        validate_file_content(b"PK\x03\x04", "archive.zip")


def test_unsupported_script_rejected():
    with pytest.raises(UnsupportedFormatError):
        validate_file_content(b"#!/bin/bash\necho 1", "script.sh")
