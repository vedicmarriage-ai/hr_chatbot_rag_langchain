from langchain_community.document_loaders import DirectoryLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.services.embedding_service import EmbeddingService
from app.core.config import get_settings
import os
import re
import shutil
from typing import List, Optional
from pathlib import Path
from langchain.schema import Document

settings = get_settings()


def clear_persist_directory(persist_directory: str) -> bool:
    """Remove an existing Chroma directory so rebuilds never append stale chunks."""
    path = Path(persist_directory)
    removed = False
    if path.exists():
        file_count = sum(1 for _ in path.rglob("*") if _.is_file())
        print(f"Removing existing vector store at {path} ({file_count} files)")
        shutil.rmtree(path)
        removed = True
    path.mkdir(parents=True, exist_ok=True)
    return removed

FAQ_QUESTION_SPLIT = re.compile(r"\n(?=Q\d+\.)")
FAQ_QUESTION_LINE = re.compile(r"^(Q\d+)\.\s*(.+)$", re.MULTILINE)


def _is_faq_section_header(line: str) -> bool:
    """Detect FAQ topic headers (Registration, Login, Profile, etc.)."""
    line = line.strip()
    if not line or len(line) > 80:
        return False
    if re.match(r"^Q\d+\.", line) or re.match(r"^Ans\.?", line, re.IGNORECASE):
        return False
    if line.startswith(("-", "Yes,", "No,")):
        return False
    if line.endswith("?"):
        return False
    if "Getting started:" in line:
        return True
    if line in (
        "Login & Password",
        "Profile Management",
        "Payment",
        "Contacting & responding to Members",
        "Searching and Filtering Profiles",
    ):
        return True
    if line.endswith("Management") and len(line) < 40:
        return True
    return False


def _faq_section_slug(section: str) -> str:
    slug = section.lower()
    slug = re.sub(r"^getting started:", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "faq"

# Article / Section / Clause boundaries (refund policy, formal legal docs)
LEGAL_BOUNDARY = re.compile(
    r"(?m)^(?:"
    r"Article\s+[IVXLC\d]+\b[^\n]*|"
    r"Section\s+\d+\b[^\n]*|"
    r"Clause\s+\d+(?:\.\d+)*\b[^\n]*|"
    r"\d+\.\d+\.\s+[^\n]+|"
    r"\d+\.\s+[A-Z][^\n]+"
    r")"
)

# Terms-style title headers (no numbering): "Privacy And Personal Data..."
TITLE_SECTION_HEADER = re.compile(
    r"(?m)^([A-Z][A-Za-z0-9 ,/&()-]{8,75})\s*$"
)

LEGAL_DOC_KEYWORDS = ("terms", "condition", "policy", "refund", "cancellation", "privacy", "legal")


class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def load_documents(self, directory: str) -> List[Document]:
        """Load documents from a directory"""
        docx_loader = DirectoryLoader(
            directory,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            exclude=["~$*"],
        )
        docx_docs = docx_loader.load()

        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        txt_docs = txt_loader.load()

        return docx_docs + txt_docs

    def _is_faq_document(self, doc: Document) -> bool:
        source = doc.metadata.get("source", "").lower()
        return "faq" in source or len(FAQ_QUESTION_SPLIT.split(doc.page_content)) >= 4

    def _split_faq_document(self, doc: Document) -> List[Document]:
        """Split FAQ into one chunk per Q&A with unique section + faq_id metadata."""
        chunks: List[Document] = []
        current_section = "FAQ"
        current_block_lines: List[str] = []

        def flush_q_block() -> None:
            nonlocal current_block_lines
            if not current_block_lines:
                return

            block = "\n".join(current_block_lines).strip()
            current_block_lines = []
            if not re.match(r"^Q\d+\.", block):
                return

            q_match = re.match(r"^(Q\d+)\.\s*", block)
            q_no = q_match.group(1) if q_match else "Q?"
            slug = _faq_section_slug(current_section)
            faq_id = f"{slug}-{q_no}"

            q_line_match = re.match(
                r"^(Q\d+\.\s*.+?)(?:\n|Ans)", block, re.DOTALL | re.IGNORECASE
            )
            faq_question = q_line_match.group(1).strip() if q_line_match else ""

            content = (
                f"{current_section}\n\n{block}" if current_section else block
            )
            chunks.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        **doc.metadata,
                        "section": current_section,
                        "chunk_type": "faq",
                        "faq_id": faq_id,
                        "faq_q_no": q_no,
                        "faq_question": faq_question,
                    },
                )
            )

        for line in doc.page_content.split("\n"):
            stripped = line.strip()

            if _is_faq_section_header(stripped):
                flush_q_block()
                current_section = stripped
                continue

            if re.match(r"^Q\d+\.", stripped):
                flush_q_block()
                current_block_lines = [line]
                continue

            if current_block_lines:
                current_block_lines.append(line)
            elif stripped and _is_faq_section_header(stripped):
                current_section = stripped

        flush_q_block()
        return chunks

    def _is_legal_document(self, doc: Document) -> bool:
        source = doc.metadata.get("source", "").lower()
        if "faq" in source:
            return False
        if any(kw in source for kw in LEGAL_DOC_KEYWORDS):
            return True
        text = doc.page_content
        return bool(LEGAL_BOUNDARY.search(text))

    def _split_legal_by_markers(self, doc: Document) -> List[Document]:
        """Split on Article / Section / numbered clause boundaries."""
        text = doc.page_content.strip()
        matches = list(LEGAL_BOUNDARY.finditer(text))
        if not matches:
            return []

        chunks: List[Document] = []
        current_section = "Introduction"

        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                chunks.append(
                    Document(
                        page_content=preamble,
                        metadata={**doc.metadata, "section": "Introduction", "chunk_type": "legal"},
                    )
                )

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if not block:
                continue

            heading = match.group(0).strip().split("\n")[0]
            is_clause = bool(re.match(r"\d+\.\d+\.", heading))
            is_article = heading.lower().startswith("article")
            is_section_only = bool(re.match(r"\d+\.\s+[A-Z]", heading)) and not is_clause

            if is_section_only and len(block) < 120:
                current_section = heading
                continue

            if is_clause or is_article:
                content = f"{current_section}\n\n{block}" if current_section != "Introduction" else block
                chunk_type = "article" if is_article else "clause"
            else:
                content = block
                current_section = heading
                chunk_type = "section"

            chunks.append(
                Document(
                    page_content=content.strip(),
                    metadata={
                        **doc.metadata,
                        "section": current_section if is_clause else heading,
                        "chunk_type": chunk_type,
                    },
                )
            )
        return chunks

    def _split_oversized_chunks(
        self, chunks: List[Document], max_chars: int = None
    ) -> List[Document]:
        """Sub-split only when a structural section still exceeds the size limit."""
        limit = max_chars or settings.CHUNK_SIZE
        result: List[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) <= limit:
                result.append(chunk)
                continue
            for i, part in enumerate(
                self.text_splitter.split_text(chunk.page_content)
            ):
                result.append(
                    Document(
                        page_content=part,
                        metadata={
                            **chunk.metadata,
                            "section_part": i + 1,
                        },
                    )
                )
        return result

    def _split_legal_by_title_headers(self, doc: Document) -> List[Document]:
        """Split terms-style docs that use title headers instead of numbering."""
        text = doc.page_content.strip()
        lines = text.split("\n")
        header_indices: List[tuple[int, str]] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("“", '"')):
                continue
            if not TITLE_SECTION_HEADER.match(stripped):
                continue
            if stripped.lower() in ("date",):
                continue
            prev_blank = i == 0 or not lines[i - 1].strip()
            next_blank = i + 1 < len(lines) and not lines[i + 1].strip()
            if not (prev_blank or next_blank):
                continue
            header_indices.append((i, stripped))

        if len(header_indices) < 2:
            return []

        chunks: List[Document] = []
        pending_header: Optional[str] = None

        if header_indices[0][0] > 0:
            preamble = "\n".join(lines[: header_indices[0][0]]).strip()
            if len(preamble) >= 80:
                chunks.append(
                    Document(
                        page_content=preamble,
                        metadata={**doc.metadata, "section": "Introduction", "chunk_type": "legal"},
                    )
                )

        for j, (line_idx, title) in enumerate(header_indices):
            end_line = (
                header_indices[j + 1][0]
                if j + 1 < len(header_indices)
                else len(lines)
            )
            block = "\n".join(lines[line_idx:end_line]).strip()
            if not block:
                continue
            if len(block) < 80:
                pending_header = title
                continue
            if pending_header:
                block = f"{pending_header}\n\n{block}"
                title = pending_header
                pending_header = None
            chunks.append(
                Document(
                    page_content=block,
                    metadata={
                        **doc.metadata,
                        "section": title,
                        "chunk_type": "section",
                    },
                )
            )
        return chunks

    def _split_legal_document(self, doc: Document) -> List[Document]:
        """Structure-aware split for policy / terms documents."""
        chunks = self._split_legal_by_markers(doc)
        if not chunks:
            chunks = self._split_legal_by_title_headers(doc)
        if not chunks:
            return self.text_splitter.split_documents([doc])
        return self._split_oversized_chunks(chunks)

    def _dedupe_chunks(self, chunks: List[Document]) -> List[Document]:
        """Drop duplicate Q&A chunks so retrieval slots are not wasted."""
        seen: set[str] = set()
        unique: List[Document] = []
        for chunk in chunks:
            key = re.sub(r"\s+", " ", chunk.page_content.strip())
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
        return unique

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents: FAQ by Q&A, legal docs by section/clause, else fixed size."""
        chunks: List[Document] = []
        for doc in documents:
            if self._is_faq_document(doc):
                chunks.extend(self._split_faq_document(doc))
            elif self._is_legal_document(doc):
                chunks.extend(self._split_legal_document(doc))
            else:
                chunks.extend(self.text_splitter.split_documents([doc]))
        return self._dedupe_chunks(chunks)

    def create_vector_store(
        self,
        documents: List[Document],
        persist_directory: str,
        embedding_function: Optional[object] = None
    ) -> Chroma:
        """Create and persist a vector store from documents"""
        if persist_directory is None:
            persist_directory = settings.CHROMA_PERSIST_DIRECTORY

        clear_persist_directory(persist_directory)

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory
        )

        vectorstore.persist()
        return vectorstore

    def process_documents(self, directory: str):
        """Complete document processing pipeline"""
        documents = self.load_documents(directory)
        splits = self.split_documents(documents)
        vectorstore = self.create_vector_store(splits)
        return vectorstore
