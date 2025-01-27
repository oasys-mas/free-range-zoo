# Installation

---

**Prerequisites:**
- `python=^3.12`
- `poetry=^1.8`
- `cargo=^1.82.0`

To install the `free-range-zoo` library:

<!--FIX: Install instructions are nonfunctional at the moment-->

```sh
# Clone the repository
git clone git@github.com:oasys-mas/free-range-zoo.git
cd free-range-zoo

# Install dependency packages
poetry install [[--with models]] # NOTE: `--with model` option is intended only for internal OASYS-MAS use.

# Verify that CUDA drivers are present and working
# NOTE: Only necessary if you are expecting to run on CUDA / GPU
python -c "import torch; print(torch.cuda.is_available())"
```


At the moment only Python `3.12` is directly supported, however it is likely that `free-range-zoo` is also compatible 
with `3.11` and `3.13`.

