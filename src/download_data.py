from pathlib import Path
from urllib.request import urlretrieve
import zipfile

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP = RAW / "online_retail_ii.zip"

print("Downloading UCI Online Retail II...")
urlretrieve(URL, ZIP)

with zipfile.ZipFile(ZIP) as z:
    z.extractall(RAW)

print("Extracted files:")
for p in RAW.iterdir():
    print(" -", p.name)
