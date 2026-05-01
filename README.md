poetry init 

poetry install

poetry run python prepare_build.py

docker compose up 

poetry run python call_psi.py --data "input_example.json" --url http://localhost:8080/predict