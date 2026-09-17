"""Install the isolated, pinned 100DOH network source and verified checkpoint."""
import hashlib
import subprocess
import sys
import urllib.request
from utils.config import load_config, resolve_path

REPOSITORY = "https://github.com/larsrpe/hand_object_detector_pytorch.git"
COMMIT = "ff6303e6e99acd710fcc36e9fba0da606cea3551"
WEIGHTS_URL = "https://huggingface.co/highbyte/frankmocap/resolve/1f9f0d3c77e74ebf1b86ad7362cebda09d82ffb3/faster_rcnn_1_8_132028.pth"
SHA256 = "3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03"


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    cfg = load_config()["contact"]
    repo, weights = resolve_path(cfg["repository_path"]), resolve_path(cfg["weights_path"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(resolve_path("requirements-contact.txt"))], check=True)
    if not repo.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected source commit {COMMIT}, found {actual}; existing checkout was left unchanged")
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        partial = weights.with_suffix(".download")
        print("Downloading archived 100DOH+ego checkpoint (378 MB)...", flush=True)
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        if checksum(partial) != SHA256:
            raise RuntimeError("Checkpoint hash mismatch; incomplete file retained with .download extension")
        partial.replace(weights)
    if checksum(weights) != SHA256:
        raise RuntimeError("Existing weights do not match the pinned mirror; not overwriting them")
    print("Contact source and checkpoint verified. Enable contact.enable_contact_detector in config/config.yaml.")


if __name__ == "__main__":
    main()
