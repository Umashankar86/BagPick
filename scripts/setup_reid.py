"""Download official Open Model Zoo ReID weights; verify upstream SHA-384."""
from hashlib import sha384
from urllib.request import urlopen
from utils.config import resolve_path

NAME = "person-reidentification-retail-0288"
BASE = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{NAME}/FP32"
CHECKSUMS = {
    "xml": "25082c8e783ad358396039d6c3969f9d862512de0cd3dd87b23c2e5790fabc64e288336e5d42369fcc4c347be15c19f2",
    "bin": "ffc5956253b60f9fac52189976a2ceba2b69288715b35caeff05238840b6e87dcc314d777b612c54d55d976988d6b9d9",
}


def main():
    directory = resolve_path("weights/reid")
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, checksum in CHECKSUMS.items():
        path = directory / f"{NAME}.{suffix}"
        if path.exists() and sha384(path.read_bytes()).hexdigest() == checksum:
            print(f"Verified: {path}")
            continue
        with urlopen(f"{BASE}/{NAME}.{suffix}", timeout=60) as response:
            data = response.read()
        if sha384(data).hexdigest() != checksum:
            raise ValueError(f"Official checksum mismatch: {suffix}")
        path.write_bytes(data)
        print(f"Downloaded and verified: {path}")


if __name__ == "__main__":
    main()
