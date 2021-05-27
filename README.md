# Code and paper for the 2021 ASME SERAD student contest


## How to build the image and enter the container

From inside the same folder as the `docker-compose.yml` file, run:
```
docker-compose build
docker-compose up -d
```
Then, enter the container with:
```
docker-compose exec terminal bash
```

## How to reproduce the results of the paper

From inside the container, run:
```
python src/train.py --model_type simple_hybrid_pqc --epochs 150 --n_runs 25 --learning_rate 0.01
```
Images will be in `docs/figures/`, numerical results will be printed to the terminal and also saved as `.csv` into `docs/results`