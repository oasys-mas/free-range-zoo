# free-range-zoo Documentation

This folder contains the documentation for [free-range-zoo](https://github.com/oasys-mas/free-range-zoo).

## Installation

To install the dependencies for the documentation package you must run the following:
```sh 
poetry install
```

Note that if you have already installed the dependencies for `free-range-zoo` then the dependencies are already available.

## Scripts
### 1a. Generating Environment Documentation Files

Here we show one direct way to build our documentation. For the following make sure you are in the `docs` directory before running these. Try below if `python -m unittest` in the `docs` directory fails to update the documentation. 

Generate the pages for each environment from environment docstrings you need to execute the following command.
```sh
poetry run gen
```

### 1b. Modifying Existing Environment Documentation Files

You can also make changes directly to the environment markdown files. From there, you can resync them with environment docstrings with the following.
```sh
poetry run sync
```

From here you have two options available:

### 2a. Building Documentation

To build the documentation files run the following:
```sh
poetry run build
```

### 2b. Watching for Changes

If you want to actively change files and watch for changes then run the following:
```sh
poetry run build
```
