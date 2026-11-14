"""Input pipeline: get plain text from pasted text, PDF or DOCX, and pull out emails/URLs."""
import io
import re

import pdfplumber
from docx import Document

EMAIL_RE = re.compile(r"[\w\.\-+]+@[\w\-]+\.[\w\.\-]+")
URL_RE = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+")


def text_from_pdf(file_obj) -> str:
    """Extract text from every page of a PDF (file-like object)."""
    with pdfplumber.open(file_obj) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def text_from_docx(file_obj) -> str:
    """Extract paragraph text from a DOCX (file-like object)."""
    doc = Document(io.BytesIO(file_obj.read()))
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(uploaded_file=None, pasted_text: str = "") -> str:
    """Return text from an uploaded file if given, otherwise from the pasted text."""
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".pdf"):
            return text_from_pdf(uploaded_file)
        if name.endswith(".docx"):
            return text_from_docx(uploaded_file)
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")
    return pasted_text or ""


def find_emails(text: str) -> list:
    return sorted(set(EMAIL_RE.findall(text)))


def find_urls(text: str) -> list:
    return sorted(set(URL_RE.findall(text)))