"""Extract and compare the text of the Phase A book .docx (task S0.7).

Used to verify book edits: extract the original and the corrected copy, then diff
them. The diff must show exactly the intended changes and nothing else.

Subcommands (run with any Python that has python-docx's dependency lxml):

    python extract_docx.py extract BOOK.docx [-o OUT.txt]
        One line per body block, in document order:
          P0123 [Style] text {link: url}        body paragraph
          T02.R03.C01.P00 [Style] text          paragraph inside table 2, row 3, cell 1
        followed by footnotes, endnotes, headers and footers.
        Text includes runs inside hyperlinks; tabs become "\\t", breaks become " | ",
        Office Math text is wrapped in \u27e6...\u27e7 so equations are visible too.

    python extract_docx.py diff ORIGINAL.docx CORRECTED.docx [--ignore-index]
        Unified diff of the two extracts (text level). --ignore-index drops body
        paragraph numbers so a deleted paragraph does not shift every later line.

    python extract_docx.py xmlcheck ORIGINAL.docx CORRECTED.docx
        Structural check at XML level:
          - every package part other than word/document.xml must be identical
            (byte-identical for binary parts, C14N-identical for XML parts;
            [Content_Types].xml may only be reordered, since entry order has no meaning);
          - body block elements are aligned and every changed block is listed, so
            a run-level edit that silently altered formatting elsewhere shows up;
          - for each replaced block, the markup with all w:t text emptied is compared,
            so "identical" means only text changed and every run kept its formatting.

    python extract_docx.py refs BOOK.docx
        Prints the reference list (paragraphs after the last "References" heading,
        up to "Non-academic links") and checks that it is in alphabetical order.

The script never writes into a .docx; it only reads.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
import zipfile

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W, "r": R, "m": M}
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _q(tag: str) -> str:
    prefix, local = tag.split(":")
    return "{%s}%s" % (NS[prefix], local)


def _load_xml(zf: zipfile.ZipFile, name: str):
    return etree.fromstring(zf.read(name))


def _hyperlink_targets(zf: zipfile.ZipFile, rels_name: str) -> dict[str, str]:
    if rels_name not in zf.namelist():
        return {}
    root = _load_xml(zf, rels_name)
    out = {}
    for rel in root.findall("{%s}Relationship" % RELS_NS):
        if rel.get("Type", "").endswith("/hyperlink"):
            out[rel.get("Id")] = rel.get("Target")
    return out


def _style_names(zf: zipfile.ZipFile) -> dict[str, str]:
    if "word/styles.xml" not in zf.namelist():
        return {}
    root = _load_xml(zf, "word/styles.xml")
    out = {}
    for st in root.findall(_q("w:style")):
        sid = st.get(_q("w:styleId"))
        name = st.find(_q("w:name"))
        out[sid] = name.get(_q("w:val")) if name is not None else sid
    return out


def paragraph_text(p) -> str:
    """Visible text of a w:p, including hyperlink runs, excluding deleted text."""
    parts = []
    for el in p.iter():
        tag = el.tag
        if tag == _q("w:t"):
            # tracked deletions use w:delText, a different tag, so they are skipped
            parts.append(el.text or "")
        elif tag == _q("m:t"):
            # Office Math (OMML) text, marked so equations are visible in the extract
            parts.append("⟦" + (el.text or "") + "⟧")
        elif tag == _q("w:tab"):
            # w:tab inside w:pPr/w:tabs is a tab-stop definition, not a character
            if el.getparent() is not None and el.getparent().tag == _q("w:tabs"):
                continue
            parts.append("\t")
        elif tag in (_q("w:br"), _q("w:cr")):
            parts.append(" | ")
        elif tag == _q("w:noBreakHyphen"):
            parts.append("‑")
    return "".join(parts).replace("\t", "\\t")


def _para_style(p, styles: dict[str, str]) -> str:
    ps = p.find("w:pPr/w:pStyle", NS)
    if ps is None:
        return "Normal"
    sid = ps.get(_q("w:val"))
    return styles.get(sid, sid)


def _para_links(p, links: dict[str, str]) -> list[str]:
    out = []
    for h in p.iter(_q("w:hyperlink")):
        rid = h.get(_q("r:id"))
        if rid and rid in links:
            out.append(links[rid])
        elif h.get(_q("w:anchor")):
            out.append("#" + h.get(_q("w:anchor")))
    return out


def _fmt_para(label: str, p, styles, links) -> str:
    line = f"{label} [{_para_style(p, styles)}] {paragraph_text(p)}"
    for url in _para_links(p, links):
        line += f" {{link: {url}}}"
    return line.rstrip()


def extract_lines(path: str) -> list[str]:
    lines: list[str] = []
    with zipfile.ZipFile(path) as zf:
        styles = _style_names(zf)
        links = _hyperlink_targets(zf, "word/_rels/document.xml.rels")
        body = _load_xml(zf, "word/document.xml").find("w:body", NS)
        p_idx = 0
        t_idx = 0
        for child in body:
            if child.tag == _q("w:p"):
                lines.append(_fmt_para(f"P{p_idx:04d}", child, styles, links))
                p_idx += 1
            elif child.tag == _q("w:tbl"):
                for r_i, tr in enumerate(child.findall("w:tr", NS)):
                    for c_i, tc in enumerate(tr.findall("w:tc", NS)):
                        for k, p in enumerate(tc.findall(".//w:p", NS)):
                            lines.append(
                                _fmt_para(f"T{t_idx:02d}.R{r_i:02d}.C{c_i:02d}.P{k:02d}", p, styles, links)
                            )
                t_idx += 1
            elif child.tag == _q("w:sdt"):
                for k, p in enumerate(child.findall(".//w:p", NS)):
                    lines.append(_fmt_para(f"SDT.P{k:03d}", p, styles, links))
        for part in sorted(zf.namelist()):
            m = re.fullmatch(r"word/(footnotes|endnotes|header\d*|footer\d*)\.xml", part)
            if not m:
                continue
            root = _load_xml(zf, part)
            for k, p in enumerate(root.iter(_q("w:p"))):
                txt = paragraph_text(p)
                if txt.strip():
                    lines.append(f"{m.group(1).upper()}.P{k:03d} {txt}")
    return lines


def _c14n(data: bytes) -> bytes:
    return etree.tostring(etree.fromstring(data), method="c14n")


def _child_set(data: bytes) -> list[bytes]:
    """Children of the root as a sorted list, for parts whose element order is not meaningful."""
    return sorted(etree.tostring(el, method="c14n") for el in etree.fromstring(data))


# Parts in which the order of the root's children carries no meaning (OPC spec);
# python-docx rewrites them in its own order on save.
_UNORDERED_PARTS = {"[Content_Types].xml"}


def _skeleton(el) -> bytes:
    """C14N of a block with every w:t text emptied: equal skeletons = text-only change."""
    import copy

    el = copy.deepcopy(el)
    for t in el.iter(_q("w:t")):
        t.text = ""
    return etree.tostring(el, method="c14n")


def xmlcheck(orig: str, new: str) -> int:
    problems = 0
    with zipfile.ZipFile(orig) as za, zipfile.ZipFile(new) as zb:
        na, nb = set(za.namelist()), set(zb.namelist())
        for name in sorted(na ^ nb):
            print(f"PART ONLY IN {'original' if name in na else 'corrected'}: {name}")
            problems += 1
        for name in sorted(na & nb):
            if name == "word/document.xml":
                continue
            a, b = za.read(name), zb.read(name)
            if a == b:
                continue
            if name.endswith(".xml") or name.endswith(".rels"):
                if _c14n(a) == _c14n(b):
                    continue
                if name in _UNORDERED_PARTS and _child_set(a) == _child_set(b):
                    print(f"part reordered only (same entries): {name}")
                    continue
            print(f"PART CHANGED: {name}")
            problems += 1
        body_a = list(_load_xml(za, "word/document.xml").find("w:body", NS))
        body_b = list(_load_xml(zb, "word/document.xml").find("w:body", NS))
        ser_a = [etree.tostring(el, method="c14n") for el in body_a]
        ser_b = [etree.tostring(el, method="c14n") for el in body_b]
        sm = difflib.SequenceMatcher(a=ser_a, b=ser_b, autojunk=False)
        changed = 0
        non_text = 0
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            changed += max(i2 - i1, j2 - j1)
            print(f"BODY BLOCKS {op}: original[{i1}:{i2}] -> corrected[{j1}:{j2}]")
            for el in body_a[i1:i2]:
                print("   - " + _short(el))
            for el in body_b[j1:j2]:
                print("   + " + _short(el))
            if op == "replace" and i2 - i1 == j2 - j1:
                for ea, eb in zip(body_a[i1:i2], body_b[j1:j2]):
                    same = _skeleton(ea) == _skeleton(eb)
                    non_text += not same
                    print("     markup apart from w:t text: " + ("identical" if same else "DIFFERS"))
        print(f"body blocks: original={len(ser_a)} corrected={len(ser_b)} changed={changed}; "
              f"replaced blocks whose markup differs beyond w:t text: {non_text}")
        print(f"non-body part problems: {problems}")
    return problems


def _short(el) -> str:
    if el.tag == _q("w:tbl"):
        first = el.find(".//w:p", NS)
        return "<table> " + (paragraph_text(first)[:80] if first is not None else "")
    return paragraph_text(el)[:110]


def _sort_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def refs(path: str) -> int:
    lines = extract_lines(path)
    start = None
    for i, line in enumerate(lines):
        # headings in this book are formatted by hand in the Normal style, so the
        # heading is found by its text; the last match skips the table of contents
        if re.match(r"P\d{4} \[[^\]]+\] (\d+\.?\s*)?(References|Bibliography)\s*$", line):
            start = i
    if start is None:
        print("no References heading found")
        return 1
    entries = []
    for line in lines[start + 1:]:
        if not line.startswith("P"):
            break
        m = re.match(r"P\d{4} \[([^\]]+)\] (.*)", line)
        if m is None:
            continue
        style, text = m.groups()
        if style.startswith("Heading") or text.strip() == "Non-academic links":
            break
        if text.strip():
            entries.append(text.strip())
    bad = 0
    for k, e in enumerate(entries):
        flag = ""
        if k and _sort_key(e) < _sort_key(entries[k - 1]):
            flag = "   <-- OUT OF ORDER"
            bad += 1
        print(f"{k:02d} {e[:100]}{flag}")
    print(f"{len(entries)} entries; out-of-order: {bad}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("docx")
    e.add_argument("-o", "--out")
    d = sub.add_parser("diff")
    d.add_argument("original")
    d.add_argument("corrected")
    d.add_argument("--ignore-index", action="store_true",
                   help="drop the body paragraph number (P0123) before diffing, so a removed "
                        "paragraph does not make every later line look changed")
    x = sub.add_parser("xmlcheck")
    x.add_argument("original")
    x.add_argument("corrected")
    r = sub.add_parser("refs")
    r.add_argument("docx")
    args = ap.parse_args(argv)

    if args.cmd == "extract":
        text = "\n".join(extract_lines(args.docx)) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)
        return 0
    if args.cmd == "diff":
        a = extract_lines(args.original)
        b = extract_lines(args.corrected)
        if args.ignore_index:
            a = [re.sub(r"^P\d{4} ", "P ", x) for x in a]
            b = [re.sub(r"^P\d{4} ", "P ", x) for x in b]
        out = list(difflib.unified_diff(a, b, "original", "corrected", n=0, lineterm=""))
        print("\n".join(out))
        n_minus = sum(1 for x in out if x.startswith("-") and not x.startswith("---"))
        n_plus = sum(1 for x in out if x.startswith("+") and not x.startswith("+++"))
        print(f"# lines removed: {n_minus}; lines added: {n_plus}")
        return 0
    if args.cmd == "xmlcheck":
        return 1 if xmlcheck(args.original, args.corrected) else 0
    if args.cmd == "refs":
        return 1 if refs(args.docx) else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
