"""Render each Supplementary Table as a standalone SVG + PNG figure.

Parses the consolidated ``supp_tables.md`` produced by
``analysis.results.supp_tables`` (or its faster-path equivalent
``C:\\Temp\\build_supp_tables.py``), splits it into per-table chunks, and
renders each chunk as a matplotlib figure with one or more tables stacked
vertically.

Output: ``figure_S_table_01.{svg,png}`` … ``figure_S_table_11.{svg,png}`` in
the same ``analysis/figures/`` folder as the other figures.

Usage::

    PYTHONPATH=analysis-repo python analysis/figures/figure_S_tables.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.figures.config import (
    cm, LABEL_SIZE, TICK_SIZE, DPI, FIGURES_DIR, save_figure,
)

# ---------------------------------------------------------------------------
# Locate the rendered supp_tables.md
# ---------------------------------------------------------------------------
PAPERS_DIR = Path(r"C:\Users\ae166\Box\CoganLab\Papers\2026\Repetition_Sub-processes")
SUPP_TABLES_MD =  "supp_tables.md"


def _split_tables(md_path: Path) -> list[tuple[int, str, str]]:
    """Return [(num, title, body), ...] for each "## Supplementary Table N." section."""
    text = md_path.read_text(encoding='utf-8')
    pattern = re.compile(
        r'^## Supplementary Table (\d+)\.\s*(.+?)$',
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Strip the trailing "---" sentinel(s) that separate Supp Table sections
        body = re.sub(r'\n---\s*\n?\s*(?:---\s*\n?)*$', '', body).strip()
        sections.append((num, title, body))
    return sections


def _extract_md_tables(body: str) -> list[tuple[str | None, list[list[str]]]]:
    """Return [(caption_or_None, rows), ...] for each markdown table in body.

    `caption_or_None` is whatever non-table line precedes the table (a
    bolded subsection header from the markdown source if present).
    """
    lines = body.splitlines()
    tables = []
    i = 0
    pending_caption = None
    while i < len(lines):
        line = lines[i]
        # A markdown table is a sequence of `| ... |` lines, where line 2 is
        # the separator `|---|---|`.
        if line.lstrip().startswith('|') and i + 1 < len(lines) \
                and re.match(r'^\s*\|[\s\-:|]+\|\s*$', lines[i + 1]):
            rows = []
            # Header
            rows.append(_split_md_row(line))
            # Skip separator
            i += 2
            # Body rows
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                rows.append(_split_md_row(lines[i]))
                i += 1
            tables.append((pending_caption, rows))
            pending_caption = None
            continue
        # Track potential caption (bold or italic single-line)
        stripped = line.strip()
        if stripped.startswith('**') and stripped.endswith('**'):
            pending_caption = stripped.strip('* ')
        elif stripped.startswith('_') and stripped.endswith('_') \
                and len(stripped) < 200:
            pending_caption = stripped.strip('_ ')
        elif stripped.startswith('### '):
            pending_caption = stripped[4:].strip()
        i += 1
    return tables


def _split_md_row(line: str) -> list[str]:
    """Split `| a | b | c |` into ['a', 'b', 'c']."""
    cells = [c.strip() for c in line.split('|')]
    if cells and cells[0] == '':
        cells = cells[1:]
    if cells and cells[-1] == '':
        cells = cells[:-1]
    # Replace Unicode glyphs that Arial lacks with ASCII equivalents
    GLYPH_MAP = {'✓': 'Y',   # check mark
                 '✗': '-',   # ballot X
                 '∈': 'in'}  # element of (only used in captions, not tables)
    out = []
    for c in cells:
        for g, repl in GLYPH_MAP.items():
            c = c.replace(g, repl)
        out.append(c)
    return out


def _extract_paragraphs(body: str) -> list[str]:
    """Return non-table paragraph snippets that are useful as caption text.

    Anything that is not part of a markdown table block; primarily
    italic / bullet / blockquote lines.
    """
    out = []
    in_table = False
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith('|') and re.match(r'^\|[\s\-:|]+\|$', s):
            continue
        if s.startswith('|'):
            in_table = True
            continue
        if in_table and not s:
            in_table = False
            continue
        if not s or s.startswith('### ') or s.startswith('---'):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_one(num: int, title: str, tables: list[tuple[str | None, list[list[str]]]],
                footer_lines: list[str]) -> plt.Figure:
    """Render one Supp Table figure with a stack of mpl tables and footer text."""
    n_tables = max(1, len(tables))
    n_rows_total = sum(len(rows) for _, rows in tables)
    # Truncate any over-long subsection caption so it doesn't wrap into the title
    tables = [(cap if cap is None or len(cap) <= 80 else cap[:77] + '...', rows)
              for cap, rows in tables]
    # Compute footer height needs (cm)
    footer_h = 0.4 * len(footer_lines) + (0.3 if footer_lines else 0)
    # Per-table caption + body height (cm)
    rows_h = 0.55 * n_rows_total + 0.7 * n_tables  # 0.7cm caption padding per table
    # Header (suptitle) height (cm)
    header_h = 1.6 + 0.25  # title block + buffer
    fig_h = max(8, header_h + rows_h + footer_h + 0.6)
    fig_w = 22
    fig = plt.figure(figsize=(fig_w * cm, fig_h * cm))

    # Title at the top, leaving plenty of room for tables below
    title_y = 1.0 - 0.2 / fig_h
    fig.suptitle(f"Supplementary Table {num}.\n{title}",
                 fontsize=LABEL_SIZE + 1, fontweight='bold', y=title_y)

    top_frac = 1.0 - header_h / fig_h
    bottom_frac = (footer_h + 0.3) / fig_h if footer_lines else 0.04

    if not tables:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        ax.text(0.5, 0.5, "(no tabular data — see catalog)",
                ha='center', va='center', fontsize=TICK_SIZE)
    else:
        gs = fig.add_gridspec(n_tables, 1,
                              hspace=0.55, top=top_frac, bottom=bottom_frac,
                              left=0.04, right=0.96)
        for t_idx, (cap, rows) in enumerate(tables):
            ax = fig.add_subplot(gs[t_idx, 0])
            ax.axis('off')
            if cap:
                ax.set_title(cap, fontsize=LABEL_SIZE, loc='left',
                             pad=6, fontweight='bold')
            header = rows[0]
            data = rows[1:]
            ncols = len(header)
            tbl = ax.table(
                cellText=data,
                colLabels=header,
                loc='center',
                cellLoc='center',
                colLoc='center',
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(TICK_SIZE)
            tbl.scale(1, 1.3)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_linewidth(0.5)
                cell.set_edgecolor('#888')
                if r == 0:
                    cell.set_facecolor('#e6eef8')
                    cell.get_text().set_fontweight('bold')
                elif r % 2 == 0:
                    cell.set_facecolor('#f7f7f7')
            tbl.auto_set_column_width(col=list(range(ncols)))
    if footer_lines:
        footer_text = '\n'.join(footer_lines)
        fig.text(0.04, 0.02, footer_text, fontsize=TICK_SIZE - 1,
                 ha='left', va='bottom', wrap=True)
    return fig


def main():
    sections = _split_tables(SUPP_TABLES_MD)
    if not sections:
        raise SystemExit(f"No Supp Table sections found in {SUPP_TABLES_MD}")

    out_dir = Path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for num, title, body in sections:
        tables = _extract_md_tables(body)
        # Footer: keep first 4 short paragraph lines (skip very long blockquote)
        paragraphs = _extract_paragraphs(body)
        footer = []
        for p in paragraphs:
            if len(p) > 280:
                continue
            footer.append(p)
            if len(footer) >= 4:
                break
        fig = _render_one(num, title, tables, footer)
        basename = f"figure_S_table_{num:02d}"
        save_figure(fig, basename, out_dir=str(out_dir),
                    dpi=DPI, exts=("svg", "png"))
        plt.close(fig)
        summary.append((num, title, len(tables), len(footer)))
        print(f"  saved {basename}.{{svg,png}} — "
              f"{len(tables)} tables, {len(footer)} footer lines",
              flush=True)

    print(f"\nDone. {len(summary)} Supp Table figures written to {out_dir}.")


if __name__ == '__main__':
    main()
