from pypdf import PdfReader
import docx

def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file.name)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = docx.Document(file.name)
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith(".txt"):
        with open(file.name, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError("Unsupported file type")
