# PettingZoo docs

This folder contains the documentation for [FreeRangeZoo](https://github.com/oasys-mas/free-range-zoo). This is adapted from [PettingZoo's Docs](https://github.com/Farama-Foundation/PettingZoo/tree/master/docs). 

## Editing an environment page

To generate the environments pages you need to execute the `docs/_scripts/gen_envs_mds.py` script:

```
cd docs
python _scripts/gen_envs_mds.py
```

## Build the Documentation

Install the required packages and PettingZoo:

```
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```

## Test the documentation
The plugin [pytest-markdown-docs](https://github.com/modal-labs/pytest-markdown-docs) allows us to test our documentation to ensure that example code runs successfully. To test, run the following command:
pytest docs --markdown-docs -m markdown-docs
