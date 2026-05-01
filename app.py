"""Flask API : probabilités et PSI vs distribution train.

Au démarrage : on charge le modèle (RUN_ID dans .env) et on calcule
les probabilités du train une fois pour toutes — c'est la distribution
de référence pour le PSI.

POST /predict   body = {"dataframe_split": {...}}
POST /psi       body = {"dataframe_split": {...}}   (uniquement actual)
"""

import io
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


sys.path.append(str(ROOT / "notebooks"))

RUN_ID = os.environ.get("RUN_ID", "local")
PORT = int(os.environ.get("PORT", 5000))
TRAIN_CSV = ROOT / "data" / "dataset.xlsx"
TARGET = 'activity'

MODEL_PATH = os.environ.get("MODEL_PATH")
if MODEL_PATH:
    MODEL = mlflow.sklearn.load_model(MODEL_PATH)
else:
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    MODEL = mlflow.sklearn.load_model(f"runs:/{os.environ['RUN_ID']}/model")

train_df = pd.read_excel(TRAIN_CSV)
TRAIN_PROBA = MODEL.predict_proba(train_df)[:, 1]

app = Flask(__name__)


def read_request_df():
    if request.content_type and "csv" in request.content_type:
        df = pd.read_csv(io.BytesIO(request.get_data()))
    else:
        s = request.get_json(force=True)["dataframe_split"]
        df = pd.DataFrame(s["data"], columns=s["columns"])
        print(df)
    return df


def proba(df):
    return MODEL.predict_proba(df)


def psi_value(expected, actual, buckets=10):
    bp = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    e = np.histogram(expected, bp)[0] / len(expected) + 1e-4
    a = np.histogram(actual, bp)[0] / len(actual) + 1e-4
    return float(np.sum((e - a) * np.log(e / a)))


@app.get("/health")
def health():
    return jsonify(status="ok", run_id=RUN_ID)


@app.post("/predict")
def predict():
    probas = proba(read_request_df())
    classes = MODEL.named_steps["classifier"].classes_
    return jsonify(
        run_id=RUN_ID,
        probabilities=[dict(zip(map(str, classes), p)) for p in probas.tolist()],
    )


@app.post("/psi")
def psi():
    return jsonify(run_id=RUN_ID, psi=psi_value(TRAIN_PROBA, proba(read_request_df())))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
