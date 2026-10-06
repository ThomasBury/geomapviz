"""Refresh the committed snapshots; gallery runs never invoke this script.

uv run --no-project --python 3.12 --with geopandas prepare_snapshots.py \
    --sources /tmp/geomapviz-gallery-source
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import geopandas as gpd
import pandas as pd


STATBEL = (
    "https://statbel.fgov.be/sites/default/files/files/opendata/"
    "Statistische%20sectoren/sh_statbel_statistical_sectors_31370_20240101.shp.zip"
)
PDOK = "https://api.pdok.nl/cbs/postcode4/ogc/v1/collections/postcode4/items"
FIELDS_BE = {
    "CNIS5_2024": "mun_id",
    "T_MUN_NL": "name_nl",
    "T_MUN_FR": "name_fr",
    "T_MUN_DE": "name_de",
    "CNIS_ARRD_": "arr_id",
    "T_ARRD_NL": "arr_nl",
    "T_ARRD_FR": "arr_fr",
    "CNIS_REGIO": "reg_id",
    "T_REGIO_NL": "reg_nl",
    "T_REGIO_FR": "reg_fr",
}
FIELDS_NL = {
    "postcode": "postcode",
    "jaarcode": "year",
    "aantal_inwoners": "residents",
    "aantal_inwoners_0_tot_15_jaar": "under15",
    "aantal_inwoners_65_jaar_en_ouder": "age65plus",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    if not path.exists():
        with urlopen(url, timeout=120) as response, path.open("wb") as target:
            shutil.copyfileobj(response, target)
    return {"url": url, "file": path.name, "sha256": digest(path)}


def write_shape(frame, target, stem):
    assert frame.crs is not None
    assert frame.geometry.is_valid.all() and not frame.geometry.is_empty.any()
    assert frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        frame.to_file(folder / f"{stem}.shp", encoding="UTF-8", index=False)
        with ZipFile(target, "w", ZIP_DEFLATED) as archive:
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                path = folder / f"{stem}{suffix}"
                # Fixed archive timestamps keep rebuilds comparable.
                info = ZipInfo(path.name, (2024, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    args.sources.mkdir(parents=True, exist_ok=True)
    output = Path(__file__).resolve().parent
    sources = [fetch(STATBEL, args.sources / "statbel-sectors-2024.zip")]
    with ZipFile(args.sources / "statbel-sectors-2024.zip") as archive:
        member = next(name for name in archive.namelist() if name.endswith(".shp"))
    sectors = gpd.read_file(
        f"zip://{args.sources / 'statbel-sectors-2024.zip'}!{member}"
    )
    assert sectors.crs.to_epsg() == 31370
    # Each municipality must have exactly one authoritative attribute tuple.
    attributes = sectors[list(FIELDS_BE)].drop_duplicates()
    assert attributes.CNIS5_2024.is_unique and not attributes.isna().any().any()
    municipalities = (
        sectors[["CNIS5_2024", "geometry"]]
        .dissolve("CNIS5_2024")
        .reset_index()
        .merge(attributes, validate="one_to_one", on="CNIS5_2024")
        .rename(columns=FIELDS_BE)
        .sort_values("mun_id")
    )
    write_shape(municipalities, output / "belgium_municipalities_2024.zip", "belgium")
    # Dissolution changes topology, but retains the sector footprint.
    footprint = sectors.geometry.union_all()
    assert footprint.symmetric_difference(municipalities.geometry.union_all()).area < 1

    url = (
        PDOK
        + "?"
        + urlencode(
            {
                "f": "json",
                "jaarcode": 2024,
                "limit": 1000,
                "crs": "http://www.opengis.net/def/crs/EPSG/0/28992",
            }
        )
    )
    features = []
    page = 1
    while url:
        path = args.sources / f"pdok-2024-page-{page:03d}.json"
        sources.append(fetch(url, path))
        content = json.loads(path.read_bytes())
        features.extend(content["features"])
        url = next(
            (link["href"] for link in content["links"] if link["rel"] == "next"), None
        )
        page += 1
    assert features and all(f["properties"]["jaarcode"] == 2024 for f in features)
    # The requested response CRS is RD New, not GeoJSON's default longitude/latitude.
    postcodes = gpd.GeoDataFrame.from_features(features, crs="EPSG:28992")
    postcodes["postcode"] = postcodes.postcode.astype(str)
    assert (
        postcodes.postcode.is_unique
        and postcodes.postcode.str.fullmatch(r"\d{4}").all()
    )
    postcodes = postcodes.sort_values("postcode")
    write_shape(
        postcodes[["postcode", "geometry"]],
        output / "netherlands_postcode4_2024.zip",
        "postcode4",
    )
    population = pd.DataFrame(postcodes[list(FIELDS_NL)].rename(columns=FIELDS_NL))
    population.to_csv(output / "netherlands_population_2024.csv", index=False)
    # Preserve raw negative suppression codes in the CSV for inspection.
    valid = population[["residents", "under15", "age65plus"]].ge(0).all(
        axis=1
    ) & population.residents.gt(0)
    licences = {
        "statbel-open-data-licence.pdf": "https://statbel.fgov.be/sites/default/files/files/opendata/Licence%20open%20data_EN.pdf",
        "cc-by-4.0.txt": "https://creativecommons.org/licenses/by/4.0/legalcode.txt",
    }
    for name, url in licences.items():
        sources.append(fetch(url, args.sources / name))
        shutil.copyfile(args.sources / name, output / name)
        if name.endswith(".txt"):
            path = output / name
            path.write_bytes(path.read_bytes().rstrip(b"\r\n") + b"\n")
    manifest = {
        "retrieved": datetime.now(timezone.utc).date().isoformat(),
        "belgium": {
            "crs": "EPSG:31370",
            "boundary_vintage": 2022,
            "sectors": len(sectors),
            "municipalities": len(municipalities),
            "arrondissements": municipalities.arr_id.nunique(),
            "regions": municipalities.reg_id.nunique(),
            "fields": FIELDS_BE,
        },
        "netherlands": {
            "crs": "EPSG:28992",
            "year": 2024,
            "postcodes": len(postcodes),
            "valid_population_cohort": int(valid.sum()),
            "excluded_population_cohort": int((~valid).sum()),
            "fields": FIELDS_NL,
        },
        "sources": sources,
        "outputs": {
            path.name: digest(path)
            for path in sorted(output.iterdir())
            if path.suffix in {".zip", ".csv", ".pdf", ".txt"}
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("belgium", "netherlands")}, indent=2))


if __name__ == "__main__":
    main()
