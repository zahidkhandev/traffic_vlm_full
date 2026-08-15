#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDES = [
    ".git/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".mypy_cache/*",
    ".ruff_cache/*",
    ".venv/*",
]


@dataclass
class PdfObject:
    obj_id: int
    data: bytes


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_text(line: str, max_chars: int) -> list[str]:
    if not line:
        return [""]
    parts: list[str] = []
    remaining = line
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    parts.append(remaining)
    return parts


def is_excluded(rel_path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def is_text_file(path: Path) -> bool:
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in raw:
        return False
    return True


def gather_files(
    root: Path,
    excludes: list[str],
    max_file_kb: int,
) -> list[Path]:
    max_bytes = max_file_kb * 1024
    collected: list[Path] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        if is_excluded(rel, excludes):
            continue
        if file_path.stat().st_size > max_bytes:
            continue
        if not is_text_file(file_path):
            continue
        collected.append(file_path)
    return collected


def build_pdf_pages(
    root: Path,
    files: list[Path],
    page_width: int = 612,   # 8.5in * 72
    page_height: int = 792,  # 11in * 72
    margin: int = 48,
    font_size: int = 10,
    line_height: int = 12,
) -> list[str]:
    max_chars = max((page_width - 2 * margin) // 6, 40)
    y_start = page_height - margin
    y_min = margin

    pages: list[str] = []
    current_lines: list[str] = []
    current_y = y_start

    def flush_page() -> None:
        nonlocal current_lines, current_y
        if not current_lines:
            return
        content = ["BT", f"/F1 {font_size} Tf"]
        y = y_start
        for raw in current_lines:
            content.append(f"1 0 0 1 {margin} {y} Tm ({escape_pdf_text(raw)}) Tj")
            y -= line_height
        content.append("ET")
        pages.append("\n".join(content))
        current_lines = []
        current_y = y_start

    def add_line(line: str) -> None:
        nonlocal current_y
        if current_y < y_min:
            flush_page()
        current_lines.append(line)
        current_y -= line_height

    header = f"Bundle: {root.as_posix()}"
    for wrapped in wrap_text(header, max_chars):
        add_line(wrapped)
    add_line("")

    for file_path in files:
        rel = file_path.relative_to(root).as_posix()
        divider = f"===== {rel} ====="
        for wrapped in wrap_text(divider, max_chars):
            add_line(wrapped)
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="latin-1")
            except OSError:
                add_line("[unreadable file]")
                add_line("")
                continue
        except OSError:
            add_line("[unreadable file]")
            add_line("")
            continue

        for raw_line in text.splitlines():
            normalized = raw_line.replace("\t", "    ")
            for wrapped in wrap_text(normalized, max_chars):
                add_line(wrapped)
        add_line("")

    flush_page()
    return pages


def write_pdf(output_path: Path, page_contents: list[str]) -> None:
    objects: list[PdfObject] = []

    font_id = 1
    content_start = 2
    content_ids = list(range(content_start, content_start + len(page_contents)))
    pages_id = content_start + len(page_contents)
    page_start = pages_id + 1
    page_ids = list(range(page_start, page_start + len(page_contents)))
    catalog_id = page_start + len(page_contents)

    objects.append(
        PdfObject(font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    )

    for content in page_contents:
        stream = content.encode("latin-1", errors="replace")
        content_obj = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects.append(PdfObject(len(objects) + 1, content_obj))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_obj = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    objects.append(PdfObject(len(objects) + 1, pages_obj))

    for cid in content_ids:
        page_obj = (
            "<< /Type /Page "
            f"/Parent {pages_id} 0 R "
            "/MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        objects.append(PdfObject(len(objects) + 1, page_obj))

    objects.append(
        PdfObject(
            len(objects) + 1,
            f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"),
        )
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(f"{obj.obj_id} 0 obj\n".encode("ascii"))
        pdf.extend(obj.data)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    total = len(objects) + 1
    pdf.extend(f"xref\n0 {total}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for i in range(1, total):
        pdf.extend(f"{offsets[i]:010} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {total} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a folder (e.g., azure_upload_bundle) into a single PDF."
    )
    parser.add_argument(
        "--input-dir",
        default="azure_upload_bundle",
        help="Input directory to package into a PDF (default: azure_upload_bundle).",
    )
    parser.add_argument(
        "--output",
        default="outputs/azure_upload_bundle.pdf",
        help="Output PDF path (default: outputs/azure_upload_bundle.pdf).",
    )
    parser.add_argument(
        "--max-file-kb",
        type=int,
        default=512,
        help="Skip files larger than this size in KB (default: 512).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude (can be provided multiple times).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    excludes = DEFAULT_EXCLUDES + list(args.exclude)
    files = gather_files(input_dir, excludes, args.max_file_kb)
    if not files:
        raise SystemExit("No readable text files found after filtering.")

    pages = build_pdf_pages(input_dir, files)
    write_pdf(output_path, pages)

    print(f"PDF created: {output_path}")
    print(f"Files included: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
