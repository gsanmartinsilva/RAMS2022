# Code and paper for the 2021 ASME SERAD student contest


## How to build the image.

```
docker build -t qmlasmeserad2021 .
```

## How to create a container from the image.

```
docker container run -it -v $PWD:/project -v $PWD/logs:/root/.guild qmlasmeserad2021 bash
```