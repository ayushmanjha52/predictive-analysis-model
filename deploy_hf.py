"""
Deploy the dashboard + API to a free Hugging Face Space (Docker SDK).

One-time:  hf auth login          (paste a token with WRITE access from
                                   https://huggingface.co/settings/tokens)
Deploy:    python deploy_hf.py    (re-run any time to push an update)

Creates the Space if it doesn't exist, uploads only what the app needs
to run (no venv, git history, raw scrape cache or model archive), and
Hugging Face builds the Dockerfile on its own servers. The app is then
live at https://<user>-<space>.hf.space/
"""
import argparse

from huggingface_hub import HfApi

SPACE_README = """---
title: Field Device Reliability
emoji: ⚙️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Combi Mill field-device reliability + power-fleet insights
---

# Field-Device Reliability: Combi Mill + Power Fleet

Delay-cause classifier, forecasts and reliability insights for the Combi
Mill's field devices, benchmarked against public power-plant data (US NRC
event reports, WRI Global Power Plant Database).

The dashboard is at `/` and the API docs are at `/docs`. Source, model card and
methodology: see the GitHub repository.
"""

IGNORE = [
    "venv/*", ".venv/*", ".git/*", "__pycache__/*", "**/__pycache__/*", "*.pyc", "logs/*",
    "catboost_info/*", ".pytest_cache/*", "models/archive/*",
    "data/external/nrc_events_raw.csv.gz", "data/external/nrc_days_done.txt",
    "Readme.md", "README.md",  # replaced by SPACE_README (Spaces need its YAML header)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="field-device-reliability", help="Space name")
    ap.add_argument("--private", action="store_true", help="make the Space private")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]  # fails clearly if not logged in
    repo_id = f"{user}/{args.space}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=".", repo_id=repo_id, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy from deploy_hf.py")
    api.upload_file(path_or_fileobj=SPACE_README.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="Space README")

    sub = repo_id.replace("/", "-").replace("_", "-").lower()
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{repo_id}")
    print(f"Live (after the ~5-10 min build): https://{sub}.hf.space/")


if __name__ == "__main__":
    main()
