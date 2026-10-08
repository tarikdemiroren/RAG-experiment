from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from local_rag.loaders import MAX_FILE_BYTES, read_docx

UPLOAD_DIR = "uploads"
MAX_UNPACKED_BYTES = 100 * 1024 * 1024
MAX_TITLE_LENGTH = 80
_UNSAFE = re.compile(r"[^\w\s\-'&(),.]")


class UploadError(ValueError):
    pass


def save_recipe(resources_dir: Path, title: str, files: list[tuple[str, bytes]]) -> Path:
    """Validate uploaded .docx files and store them as ``<resources>/uploads/<title>/``."""
    name = safe_title(title)
    if not files:
        raise UploadError("Add at least one .docx file.")
    texts = [_read(filename, data) for filename, data in files]
    if not any(texts):
        raise UploadError("The documents contain no text.")

    root = (resources_dir / UPLOAD_DIR).resolve()
    target = root / name
    if target.parent != root:
        raise UploadError("Invalid recipe name.")
    if target.exists():
        raise UploadError(f"A recipe named {name!r} already exists.")

    root.mkdir(parents=True, exist_ok=True)
    # Write into a hidden folder first so a failed upload never leaves a half-written recipe.
    staging = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    try:
        for i, (_, data) in enumerate(files, 1):
            (staging / f"{i}.docx").write_bytes(data)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def safe_title(title: str) -> str:
    name = " ".join(_UNSAFE.sub("", title).split())[:MAX_TITLE_LENGTH].strip(" .")
    if not name:
        raise UploadError("Enter a recipe name.")
    return name


def _read(filename: str, data: bytes) -> str:
    if not filename.lower().endswith(".docx"):
        raise UploadError(f"{filename}: only .docx files are supported.")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"{filename}: larger than {MAX_FILE_BYTES // 2**20} MB.")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unpacked = sum(info.file_size for info in archive.infolist())
        if unpacked > MAX_UNPACKED_BYTES:
            raise UploadError(f"{filename}: unpacks to more than {MAX_UNPACKED_BYTES // 2**20} MB.")
        return read_docx(io.BytesIO(data))
    except UploadError:
        raise
    except Exception:
        raise UploadError(f"{filename}: not a valid .docx file.") from None
