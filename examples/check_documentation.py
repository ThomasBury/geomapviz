"""Check the built geographic gallery and its downloadable bundle (stdlib only).

Run after generating the bundle and building Zensical:
    python examples/check_documentation.py
"""

import argparse
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile


class Page(HTMLParser):
    def __init__(self, filename):
        super().__init__()
        self.links = []
        self.frames = []
        self.text = []
        self.feed(filename.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src", "data-src"):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == "iframe":
            assert attrs.get("title"), "Every interactive embed needs a title"
            assert attrs.get("loading") == "lazy"
            assert "src" not in attrs, "Closed embeds must not load large HTML files"
            self.frames.append(attrs["data-src"])

    def handle_data(self, data):
        self.text.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args()
    site = args.site.resolve()
    root = Path(__file__).resolve().parent.parent
    source = (root / "examples/geographic_gallery.py").read_text()
    tutorials = ("belgium", "aggregation", "rates", "netherlands", "demographics")
    for name in tutorials:
        filename = site / "examples" / name / "index.html"
        page = Page(filename)
        text = "".join(page.text)
        section = source.split(f"# --8<-- [start:{name}]\n", 1)[1]
        section = section.split(f"# --8<-- [end:{name}]", 1)[0].strip()
        assert section in text, f"Canonical source snippet missing: {name}"
        assert "--8<--" not in text
        assert len(page.frames) == 1
    for filename in [
        site / "index.html",
        *site.glob("*/index.html"),
        *site.glob("examples/*/index.html"),
    ]:
        for link in Page(filename).links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            target = (filename.parent / unquote(url.path)).resolve()
            assert target.is_relative_to(site), (filename, link)
            assert target.exists(), (filename, link)
    for preview in (root / "docs/assets/geographic").iterdir():
        assert (
            site / "assets/geographic" / preview.name
        ).read_bytes() == preview.read_bytes()
    with ZipFile(site / "downloads/geographic-examples.zip") as bundle:
        assert bundle.testzip() is None
        names = {n for n in bundle.namelist() if not n.endswith("/")}
        assert len(names) == 11, names
        for name in names:
            assert bundle.read(name) == (root / "examples" / name).read_bytes(), name
        for name in ("belgium_municipalities_2024", "netherlands_postcode4_2024"):
            with ZipFile(BytesIO(bundle.read(f"data/{name}.zip"))) as shape:
                assert {Path(n).suffix for n in shape.namelist()} == {
                    ".shp",
                    ".shx",
                    ".dbf",
                    ".prj",
                    ".cpg",
                }
    print("Gallery source snippets, links, titled embeds, previews and bundle pass")


if __name__ == "__main__":
    main()
