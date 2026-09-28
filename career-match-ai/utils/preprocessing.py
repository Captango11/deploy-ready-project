"""NLP preprocessing: PDF text extraction, normalization, technical-term
preservation and tokenization."""
import io
import re

# Technical terms that normal punctuation stripping would destroy.
# Each is mapped to a safe placeholder token before cleaning.
PROTECTED_TERMS = [
    (r"c\+\+", " cplusplus "),
    (r"c#", " csharp "),
    (r"f#", " fsharp "),
    (r"\.net\b", " dotnet "),
    (r"node\.js|nodejs", " nodejs "),
    (r"vue\.js|vuejs", " vuejs "),
    (r"next\.js|nextjs", " nextjs "),
    (r"react\.js|reactjs", " react "),
    (r"express\.js|expressjs", " express "),
    (r"ci\s*/\s*cd", " cicd "),
    (r"a\s*/\s*b testing", " ab_testing "),
    (r"github actions", " github_actions "),
    (r"rest(ful)?\s*apis?", " rest_api "),
    (r"scikit[- ]learn|sklearn", " scikit_learn "),
    (r"power\s*bi", " power_bi "),
    (r"spring\s*boot", " spring_boot "),
]

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
NON_TOKEN_RE = re.compile(r"[^a-z0-9_\s]")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = set(
    """a an the and or of to in for on with at by from as is are was were be been
    i me my we our you your he she it its they them this that these those have has
    had do does did so but if then than into over about also where which who
    while during can will would should could""".split()
)


class PDFExtractionError(Exception):
    """Raised with a user-friendly message when a PDF cannot be read."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Step 1: extract raw text from a PDF using PyPDF2."""
    from PyPDF2 import PdfReader

    if not file_bytes:
        raise PDFExtractionError("The uploaded PDF is empty.")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise PDFExtractionError("The PDF is password-protected.")
        if len(reader.pages) == 0:
            raise PDFExtractionError("The PDF has no pages.")
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
    except PDFExtractionError:
        raise
    except Exception:
        raise PDFExtractionError("The PDF appears to be corrupted or unreadable.")
    if len(text.strip()) == 0:
        raise PDFExtractionError(
            "No extractable text found. The PDF may be a scanned image — "
            "please paste the resume text instead."
        )
    return text


def normalize(text: str) -> str:
    """Steps 2-3: lowercase, remove URLs/emails, preserve technical terms,
    strip irrelevant symbols and normalize whitespace."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    for pattern, repl in PROTECTED_TERMS:
        text = re.sub(pattern, repl, text)
    text = NON_TOKEN_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    """Step 4: tokenize normalized text."""
    tokens = TOKEN_RE.findall(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1 or t in {"r", "c"}]
    return tokens


def preprocess(text: str) -> str:
    """Full NLP pipeline -> cleaned token string ready for TF-IDF."""
    return " ".join(tokenize(normalize(text)))


def split_sentences(raw_text: str) -> list:
    """Split raw resume text into sentences / bullet lines (for evidence)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|•|▪|●", str(raw_text))
    return [WS_RE.sub(" ", p).strip(" -*\t") for p in parts if len(p.strip()) > 15]
