"""Prépare le contexte Docker en isolant les artifacts du RUN_ID cible.

Usage : poetry run python prepare_build.py
        (lit RUN_ID depuis .env)

Crée ./build_model/ contenant uniquement les artifacts du run cible,
ce qui permet au Dockerfile de copier juste ce dossier — peu importe
combien d'expérimentations sont stockées dans mlruns/.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

RUN_ID = os.environ["RUN_ID"]
MLRUNS = ROOT / "mlruns"
DEST = ROOT / "build_model"


def find_model_dir(run_id: str) -> Path:
    for meta in MLRUNS.glob("*/models/*/meta.yaml"):
        if f"source_run_id: {run_id}" in meta.read_text():
            return meta.parent
    raise SystemExit(f"Aucun modèle trouvé pour RUN_ID={run_id}")


model_dir = find_model_dir(RUN_ID)
artifacts = model_dir / "artifacts"

if DEST.exists():
    shutil.rmtree(DEST)
shutil.copytree(artifacts, DEST)

print(f"OK : {artifacts}  ->  {DEST}", file=sys.stderr)
