# Installation

In this guide we walk through installing [free-range-zoo](https://github.com/oasys-mas/free-range-zoo). For this we 
need a python virtual environment, here we use miniconda. The action/observation spaces in free-range-zoo rely on 
rust functions implemented in [free-range-rust](https://github.com/C4theBomb/free-range-rust) for efficency. Compiling 
these is part of the `poetry` install, so we will: 
 
1. Install gcc-13 / clang and [Rust](https://www.rust-lang.org/tools/install)
2. Clone  [free-range-zoo](https://github.com/oasys-mas/free-range-zoo)
3. Install [Miniconda](https://docs.anaconda.com/miniconda/install/)
4. Create a python **3.12** environment, we support no lower version
5. Install dependencies
6. Test

> We have experienced problems with installation in macOS computers which utilize x86_64 architectures. `pytorch` no 
longer supports x86 macOS.

## Install gcc-13 / clang

> If `gcc --version` >= 13.0, continue to the next step. We set these paths so `maturin`, the rust<->python binder, 
uses the right version of gcc.

### Linux (GCC 13)

-  Install gcc
   - If in **Ubuntu** `sudo apt install gcc-13`
   - for **fedora/rhel/CentOS** `sudo dnf install gcc-toolset-13`
- Set your `gcc` and `g++` paths. Installing gcc with (2) will install `g++` as well.
   - `export CC="<path to your gcc v13 binary>"`
   - `export CXX="<path to your g++ v13 binary>`
- Make sure the version is correct
   - `gcc -v` = `13.x`

### MacOS (Clang) - aarch64

> `brew install llvm@15`: Install clang 14/15
- Run the path override command that homebrew gives you when running this (this updates your `.zshrc` or `.bashrc`).
>`gcc -v`: Make sure the version is correct and `^15.0`

### Windows (MinGW-w64)

- Install [Chocolatey](https://chocolatey.org/install)
- Install GCC/MinGW (in administrative powershell)
  - `choco install mingw` currently this gives the right version. You may have to specify version in the future.
- Add `mingw` to path. 
  - Search for `environment variables` in the windows search bar. Click on "environment variables" Then click on `path` in the list. Click `edit`. See if there is a entry for `mingw` If not find the `mingw64\mingw64\bin` folder and add it to `path` (typically this is in `C:\ProgramData`).
- Restart powershell
- Make sure the version is correct
-   `gcc -v` >= `13.x`
-   Reboot your machine. (Needed for rust install if you just installed chocolatey)

## Rust

### Linux / MacOS
Go to [Rust Install](https://www.rust-lang.org/tools/install) select your os, and run the curl command.

### Windows
> `choco install rust`: Install Rust, restart machine to see effects.

> `rustc --version`: Verify that the RustC compiler was installed.

> `cargo --version`: Verify that cargo was properly installed.

## Clone Free-Range-Zoo

<!--TODO: Correct this URL to the competition repository.-->
> `git clone git@github.com:oasys-mas/free-range-zoo.git`: Clone the latest release of the free-range-zoo repository.

## Miniconda

> ⚠ WARNING: DO NOT USE SUDO HERE! IT IS VERY LIKELY TO BREAK INSTALL.

- Run the installation script for [Miniconda](https://docs.anaconda.com/miniconda/install/), select your os. 



- Install with defaults, and say `yes` to the conda init question.
- Restart your terminal
- You should see something like `(base) user@machine:...` 

>If you don't then double check that the `miniconda3` folder is in your home directory and that it was added to your path.

## Create Python env

> `conda create -n three12 python=3.12`: Create a python 3.12 environment
> `conda activate three12`: Activate the created python environment

## Install Dependencies

> `pip install poetry`: Install poetry, the dependency tool we use.
> `poetry install`: Install all dependencies with poetry

## Test

> `python -m unittest -b`: To see if everything is working try running our unit tests. This will build a local copy of 
our docs page, see `free-range-zoo/docs/build/html/index.html`.

### Windows

When on Windows, the unittest will not build the docs. To build them:

> If you are having trouble getting `free-range-rust` to install, and you still want to build the docs, run `pip install -e .` in the root of the `free-range-zoo` repository before doing this. Then **uninstall the frz library first before reattempting to install free-range-rust** `pip uninstall free-range-zoo`.

-  navigate to `free-range-zoo/docs`. 
-  Ensure you are on the `three12` conda environment
-  run  `python free_range_zoo_docs/gen_envs_docstrings.py`
-  run `python free_range_zoo_docs/gen_envs_mds.py`
-  run `make build`.  
