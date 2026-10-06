# Frozen geographic data

These files belong to the downloadable examples, not the installed package.
Running the gallery reads only local files. No download, tile service or Python
server is needed. Geometry is retained at full source detail, without simplification.

Retrieved **6 October 2026**. [manifest.json](manifest.json) records the exact
source URLs, SHA-256 checksums, output checksums, field mappings and row counts.
Checksums identify this snapshot, rather than guaranteeing that upstream URLs
will continue to return the same bytes.

## Belgium

Source: **Statistics Belgium (Statbel), statistical sectors on 1 January 2024**,
[dataset and metadata](https://statbel.fgov.be/en/open-data/statistical-sectors-2024).
The underlying municipality boundary vintage is **2022**, as stated by Statbel.
This is a historical teaching snapshot; it does not describe later municipal mergers.

`belgium_municipalities_2024.zip` contains `belgium.shp`, `.shx`, `.dbf`, `.prj`
and `.cpg` (UTF-8). It dissolves all **19,795 sectors** into **581 municipalities**
in **EPSG:31370 (Belgian Lambert 1972)**. The source footprint is preserved.
Each municipality has one explicit arrondissement and region mapping:
**43 arrondissements**, **3 regions**. Never infer these codes from sector IDs;
Statbel warns that this is no longer valid after the 2019 changes.

| Source field | Snapshot field | Meaning |
| --- | --- | --- |
| `CNIS5_2024` | `mun_id` | Municipality code, stored as text |
| `T_MUN_NL`, `T_MUN_FR`, `T_MUN_DE` | `name_nl`, `name_fr`, `name_de` | Municipality names |
| `CNIS_ARRD_` | `arr_id` | Authoritative arrondissement code |
| `T_ARRD_NL`, `T_ARRD_FR` | `arr_nl`, `arr_fr` | Arrondissement names |
| `CNIS_REGIO` | `reg_id` | Authoritative region code, including leading zero |
| `T_REGIO_NL`, `T_REGIO_FR` | `reg_nl`, `reg_fr` | Region names |

Redistribution and derivation follow the included
[Statbel open data licence](statbel-open-data-licence.pdf). Attribution:
Statistics Belgium, statistical sectors 2024, municipality boundaries 2022;
municipality dissolution by the Geomapviz documentation contributors.
Statbel does not endorse these tutorials. All Belgian tutorial measurements
and predictions are simulated, not Statbel population observations.

## Netherlands

Source: **Centraal Bureau voor de Statistiek (CBS), postcode4 statistics 2024**,
published through [PDOK](https://api.pdok.nl/cbs/postcode4/ogc/v1?f=html&lang=en).
The API response is filtered with **`jaarcode=2024`**, follows every `next` link,
and requests **EPSG:28992 (RD New)** coordinates. The API's year-specific
geometry is retained without repair, rounding or simplification.

`netherlands_postcode4_2024.zip` contains `postcode4.shp`, `.shx`, `.dbf`, `.prj`
and `.cpg` (UTF-8), with **4,071 unique four-digit postcode polygons**. The
`postcode` identifier is text. `netherlands_population_2024.csv` contains one
row per boundary and preserves the published integer counts and suppression codes:

| Source field | CSV field | Meaning |
| --- | --- | --- |
| `postcode` | `postcode` | Four-digit postcode, read explicitly as text |
| `jaarcode` | `year` | Reference year, 2024 |
| `aantal_inwoners` | `residents` | Registered residents on 1 January |
| `aantal_inwoners_0_tot_15_jaar` | `under15` | Residents younger than 15 |
| `aantal_inwoners_65_jaar_en_ouder` | `age65plus` | Residents aged 65 or older |

CBS's [publication notes, sections 4.3 and 4.6](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers)
explain rounding and suppression. Counts are rounded to multiples of five.
**-99997 means 0–4 / suppressed / absent; -99995 means not yet published.**
Neither code is a measured zero. The runner treats all negative counts as
missing, then selects one common cohort with three available counts and a
positive resident denominator. This leaves **3,968 valid postcodes** and **103
excluded postcodes**, whose boundaries remain visible in grey. Percentages are
calculated as `100 * age_count / residents`; they are approximate ratios of
rounded counts, not CBS's separately published rounded percentage fields.
They describe registered residents, not everyone physically present in an area.

Attribution: **CBS / PDOK, postcode4 statistics 2024**, retrieved 6 October 2026,
selected and converted to shapefile/CSV by the Geomapviz documentation contributors.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
the [licence text](cc-by-4.0.txt) is included. CBS and PDOK do not endorse this work.
Only the licence text's trailing blank line is normalized; its wording is retained.

## Rebuild the snapshots

The committed snapshots are sufficient to run the examples. To repeat their
preprocessing, run from this directory:

```sh
uv run --no-project --python 3.12 --with geopandas prepare_snapshots.py \
  --sources /tmp/geomapviz-gallery-source
```

This maintenance command needs internet access to fetch the sources and licences.
It caches the original Statbel ZIP and each full PDOK response page in the
specified directory. For byte-identified inputs, retain those cached files and
compare their checksums against `manifest.json`; upstream revisions may otherwise
produce a different snapshot. The script checks year selection, unique IDs,
complete administrative mappings, polygon validity, CRS and the Statbel footprint.
It dissolves only Statbel sector geometry, retains one verified municipality
attribute tuple, writes complete shapefiles and records provenance. No geometry
simplification pipeline is used.
