"""
Publishes site/ (built by build_site.py) to the gh-pages branch, which
GitHub Pages serves at https://ayushmanjha52.github.io/predictive-analysis-model/

Run:  python src/build_site.py && python src/publish_site.py
"""
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    if not (SITE / "index.html").exists():
        sys.exit("site/ not built -- run `python src/build_site.py` first.")
    remote = git("remote", "get-url", "origin")
    name, email = git("config", "user.name"), git("config", "user.email")
    source = git("rev-parse", "--short", "HEAD")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "site"
        shutil.copytree(SITE, work)
        git("init", "-q", "-b", "gh-pages", cwd=work)
        git("config", "user.name", name, cwd=work)
        git("config", "user.email", email, cwd=work)
        git("add", "-A", cwd=work)
        git("commit", "-q", "-m", f"Publish site from {source} ({datetime.now():%Y-%m-%d})", cwd=work)
        git("push", "-q", "--force", remote, "gh-pages", cwd=work)
    print("Published to gh-pages: https://ayushmanjha52.github.io/predictive-analysis-model/")


if __name__ == "__main__":
    main()
