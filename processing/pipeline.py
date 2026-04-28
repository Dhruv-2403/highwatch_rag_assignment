from __future__ import annotations

import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field

import PyPDF2
import docx  # python-docx

from config import settings
from connectors.gdrive import DriveFile

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
TXT_MIME = "text/plain"


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""

    chunk_id: str         
    doc_id: str           
    file_name: str
    source: str = "gdrive"
    text: str = ""
    chunk_index: int = 0
    web_view_link: str = ""
    modified_time: str = ""
    extra_metadata: dict = field(default_factory=dict)



def _extract_pdf(content: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _extract_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_txt(content: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def extract_text(drive_file: DriveFile) -> str:
    mime = drive_file.mime_type
    try:
        if mime == PDF_MIME:
            return _extract_pdf(drive_file.content)
        if mime == DOCX_MIME:
            return _extract_docx(drive_file.content)
        if mime == TXT_MIME:
            return _extract_txt(drive_file.content)
        logger.warning("Unsupported MIME %s for %s", mime, drive_file.file_name)
        return ""
    except Exception as exc: 
        logger.error("Extraction failed for %s: %s", drive_file.file_name, exc)
        return ""


def clean_text(raw: str) -> str:

    text = unicodedata.normalize("NFKC", raw)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()



def _approx_tokens(text: str) -> int:

    return len(text) // 4


def _split_into_paragraphs(text: str) -> list[str]:
  
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    
    chunk_size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

 
    char_limit = chunk_size * 4
    overlap_chars = overlap * 4

    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
      
        if len(para) > char_limit:
     
            if current:
                chunks.append("\n\n".join(current))
      
                tail = "\n\n".join(current)[-overlap_chars:]
                current = [tail] if tail.strip() else []
                current_len = len(tail)

            
            for i in range(0, len(para), char_limit - overlap_chars):
                segment = para[i : i + char_limit].strip()
                if segment:
                    chunks.append(segment)
            continue

        if current_len + len(para) + 2 > char_limit and current:
            chunks.append("\n\n".join(current))
         
            tail = "\n\n".join(current)[-overlap_chars:]
            current = [tail.strip()] if tail.strip() else []
            current_len = len(current[0]) if current else 0

        current.append(para)
        current_len += len(para) + 2 

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


# ── Public pipeline ──────────────────────────────────────────────────────────

def process_drive_file(drive_file: DriveFile) -> list[Chunk]:
    #  extract - clean - chunk the processing pipeline
    raw = extract_text(drive_file)
    if not raw:
        logger.warning("No text extracted from %s", drive_file.file_name)
        return []

    cleaned = clean_text(raw)
    texts = chunk_text(cleaned)

    chunks: list[Chunk] = []
    for idx, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"{drive_file.file_id}_{idx}",
                doc_id=drive_file.file_id,
                file_name=drive_file.file_name,
                source="gdrive",
                text=text,
                chunk_index=idx,
                web_view_link=drive_file.web_view_link,
                modified_time=drive_file.modified_time,
            )
        )

    logger.info(
        "Processed '%s': %d chars → %d chunks",
        drive_file.file_name,
        len(cleaned),
        len(chunks),
    )
    return chunks
