"""Extract visible text and speaker notes from the final PPTX; no edits."""
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/presentations"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def paragraphs(root):
    return [text for para in root.findall(".//a:p", NS)
            if (text := "".join(t.text or "" for t in para.findall(".//a:t", NS)))]


def target_path(part, target):
    return target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(posixpath.dirname(part), target))


def main():
    visible = ["# Slide text — relational memory v12\n\nSlides 1–14 are the main presentation. Slides 15–21 are backups. Text below comes from the PPTX; chart values are available in the editable charts and numerical evidence.\n"]
    notes = ["# Presenter guide — relational memory v12\n\nStart with [the team catch-up](../../docs/TEAM_CATCHUP.md). It explains the pinwheel example, score scales, cutoff crossings, implementation and remaining experiments. The following are the actual speaker notes from the final PPTX. Main slides use proposal language.\n"]
    with ZipFile(OUT / "relational-memory-proposal-v12.pptx") as archive:
        slides = sorted((p for p in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", p)), key=lambda p: int(re.search(r"(\d+)\.xml", p).group(1)))
        for number, part in enumerate(slides, 1):
            root = ET.fromstring(archive.read(part))
            relpart = posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels")
            relationships = ET.fromstring(archive.read(relpart))
            rels = {r.attrib["Id"]: r.attrib for r in relationships}
            lines = [s for s in paragraphs(root) if s.strip() not in {str(number), f"Backup  {number}"}]
            title = lines[0]
            visible.append(f"## Slide {number}: {title}\n\n" + "\n\n".join(lines[1:]))
            for chart in root.findall(".//c:chart", NS):
                rel = rels[chart.attrib["{" + NS["r"] + "}id"]]
                content = ET.fromstring(archive.read(target_path(part, rel["Target"])))
                labels = []
                for ser in content.findall(".//c:ser", NS):
                    tx = ser.find("c:tx", NS)
                    if tx is not None:
                        label = " ".join(v.text or "" for v in tx.findall(".//c:v", NS))
                        if label:
                            labels.append("Series: " + label)
                for axis in content.findall(".//c:valAx", NS) + content.findall(".//c:catAx", NS):
                    ax_title = axis.find("c:title", NS)
                    if ax_title is not None:
                        labels.append("Axis: " + " ".join(paragraphs(ax_title)))
                visible.append("\nChart labels:\n\n" + "\n".join("- " + value for value in labels))
            for rel in rels.values():
                if rel["Type"].endswith("/notesSlide"):
                    nr = ET.fromstring(archive.read(target_path(part, rel["Target"])))
                    body = []
                    for shape in nr.findall(".//p:sp", NS):
                        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
                        if ph is not None and ph.get("type") in {"sldNum", "hdr", "ftr", "dt"}:
                            continue
                        body.extend(paragraphs(shape))
                    notes.append(f"## Slide {number}: {title}\n\n" + "\n\n".join(body))
    (OUT / "slide-text-v12.md").write_text("\n\n".join(visible) + "\n")
    (OUT / "presenter-guide-v12.md").write_text("\n\n".join(notes) + "\n")
    print(f"Exported {len(slides)} slides and their notes.")


if __name__ == "__main__":
    main()
