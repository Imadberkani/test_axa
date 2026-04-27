"""Appelle l'API en envoyant un fichier (CSV ou JSON).

Format détecté automatiquement par l'extension du fichier.

Usage
-----
    poetry run python call_psi.py --data data/cs-test.csv --url http://localhost:8080/predict
    poetry run python call_psi.py --data input_example.json --url http://localhost:8080/predict
"""

import click
import requests

CONTENT_TYPES = {".csv": "text/csv", ".json": "application/json"}


@click.command()
@click.option("--data", "path", required=True, help="Chemin du fichier à envoyer (.csv ou .json).")
@click.option("--url", default="http://localhost:8080/psi", show_default=True)
def main(path, url):
    ext = "." + path.rsplit(".", 1)[-1].lower()
    content_type = CONTENT_TYPES.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        r = requests.post(url, data=f, headers={"Content-Type": content_type})
    click.echo(f"{r.status_code} {r.text[:500]}")


if __name__ == "__main__":
    main()
