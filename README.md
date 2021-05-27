# Code and paper for the 2021 ASME SERAD student contest

## How to build the image and enter the container
From the same folder as the `Dockerfile` file, run:

```
docker build . -t asmeserad2021
```

Then, create and enter a container

```
docker run -it -v $PWD/src:/project/src -v $PWD/data:/project/data -v $PWD/docs:/project/docs asmeserad2021 bash
```


## How to reproduce the results of the paper

Go into the `src` folder:

```
cd src/
```

From inside the container, run:

```
python train.py --model_type simple_hybrid_pqc --epochs 150 --n_runs 25 --learning_rate 0.01
```

Figures will be in `docs/figures/`, numerical results will be printed to the terminal and also saved as `.csv` into `docs/results/`