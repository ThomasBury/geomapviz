import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

EXTRAS_REQUIRE = {"tests": ["pytest", "pytest-cov"]}

INSTALL_REQUIRES = [
    "pandas >= 1.0.0",
    "numpy >= 1.18.0",
    "geopandas >= 0.8.0",
    "mapclassify >= 2.4.0",
    "holoviews >= 1.14.0",
    "geoviews >= 1.8.1",
    "cartopy >= 0.17.0",
    "cmasher >= 1.5.8",
    "palettable >= 3.3.0",
    "matplotlib >= 3.3.0",
    "seaborn >= 0.11.0",
]

KEYWORDS = "geographical, visualization, map, interactive, choropleth"

setup(
    name="geomapviz",
    version="0.6.2",
    description="Geographical visualization",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ThomasBury/geomapviz",
    author="Thomas Bury",
    author_email="bury.thomas@gmail.com",
    packages=find_packages(),
    zip_safe=False,  # the package can run out of an .egg file
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.6",
    license="MIT",
    keywords=KEYWORDS,
    package_data={'': ['beshp/*.cpg', 'beshp/*.dbf', 'beshp/*.shp', 'beshp/*.shx']},
)
