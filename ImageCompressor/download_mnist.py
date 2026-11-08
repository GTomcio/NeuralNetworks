from pathlib import Path
from urllib.request import urlretrieve
import gzip
import shutil

data_dir = Path(__file__).parent / "data" / "MNIST" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

for filename in files:
    compressed = data_dir / filename
    extracted = data_dir / filename.removesuffix(".gz")

    if not extracted.exists():
        print(f"Downloading {filename}...")
        urlretrieve(base_url + filename, compressed)

        with gzip.open(compressed, "rb") as source, extracted.open("wb") as target:
            shutil.copyfileobj(source, target)

        compressed.unlink()

print(f"MNIST files are ready in: {data_dir}")