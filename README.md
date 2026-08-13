# ml_tuning

β-VAE + flow-matching training (`latentflow.py`), driven by a CAF actor
sweep (`actors/test.cc`) that fans a list of β values out across worker
actors and, per job, round-robins onto the local GPUs.

## Prerequisites: installing the CAF framework

`test_actor` is built against **CAF (the C++ Actor Framework)**, and it
needs the **>=1.0 API** (`anon_mail`, `self->println`, `self->state()`).
Most systems only ship an older CAF (e.g. `/usr/local/caf` is 0.18.6),
which doesn't have these, so you'll likely need to build CAF yourself
from source. It's a quick, one-time setup.

**1. Grab the source**

```bash
git clone https://github.com/actor-framework/actor-framework.git
cd actor-framework
```

**2. Build and install it somewhere of your choosing**

Pick any install location you like — a folder in your home directory
works well so you don't need root access:

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/caf-install \
  -DCAF_ENABLE_EXAMPLES=OFF -DCAF_ENABLE_TESTING=OFF
cmake --build build -j$(nproc)
cmake --install build
```

**3. Confirm where the libs landed**

Depending on your distro, the install step may place libraries under
`lib/` or `lib64/`:

```bash
ls $HOME/caf-install
```

Note both the install prefix (`$HOME/caf-install` above) and that
lib folder name — you'll need them in the next step.

## Building `test_actor`

The [Makefile](actors/Makefile) points at a specific CAF install by
default (`CAF_PREFIX`/`CAF_LIBDIR`), which almost certainly won't match
where *you* installed CAF. Point it at your own install with:

```bash
cd actors
make CAF_PREFIX=$HOME/caf-install CAF_LIBDIR=lib
```

(swap in whatever prefix and lib folder you noted in step 3 above — use
`lib64` there if that's what `ls` showed).

If you'd rather not type the override every time, edit the defaults at
the top of [`actors/Makefile`](actors/Makefile) to match your install
path:

```make
CAF_PREFIX ?= $(HOME)/caf-install
CAF_LIBDIR ?= lib
```

Then a plain `make` will pick it up automatically.

## Python environment

This project was developed and run using a Python virtual environment
(`.venv`) at the repo root — that's what `../.venv/bin/python3` refers to
below. If you'd rather use a system Python (or your own venv/conda env)
that already has the required libraries installed, you'll need to swap
that path out in a couple of places instead of creating a `.venv`:

- **`actors/test.cc`** (line 59) — the actor sweep hardcodes
  `../.venv/bin/python3` when it shells out to `latentflow.py`. Change it
  to point at your interpreter (e.g. `python3`, or the full path from
  `which python3`), then rebuild with `make`.
- **The direct-run command** in
  ["Running `latentflow.py` directly"](#running-latentflowpy-directly)
  below — swap `../.venv/bin/python3` for whichever interpreter has your
  libraries installed.


## Running a local sweep (server mode)

Server mode spawns worker actors in-process, hands each one a β value from
the sweep, and prints `beta = ..., MMD = ...` as each job finishes. Run it
from `actors/` :

```bash
cd actors
./test_actor --server-mode --betas 0.1,0.5,1.0
```

Useful flags (see `config.h`):

| Flag | Default | Meaning |
|---|---|---|
| `--betas`, `-b` | `0.1,0.5,1.0,2.0,4.0` | comma-separated β values to sweep |
| `--workers`, `-w` | `3` | parallel worker actors (each pulls the next queued β when it finishes one) |
| `--gpus`, `-g` | `2` | local GPUs to round-robin jobs across (`0` disables pinning) |
| `--port`, `-p` | `0` | port the server publishes itself on |

Examples:

```bash
# Default sweep (0.1, 0.5, 1.0, 2.0, 4.0), 3 workers, 2 GPUs
./test_actor --server-mode

# Custom beta values, e.g. a coarse sweep of 5 values
./test_actor --server-mode --betas 0.1,0.5,1.0,2.0,4.0

# More workers than GPUs is fine -- jobs round-robin across --gpus GPUs
./test_actor --server-mode --betas 0.1,0.25,0.5,0.75,1.0,2.0 --workers 5 --gpus 2

# Single beta value
./test_actor --server-mode --betas 1.0 --workers 1
```

Or via the Makefile (`ARGS` is passed straight through to `test_actor`):

```bash
make run ARGS="--server-mode --betas 0.1,0.5,1.0 --workers 3 --gpus 2"
```

Each job writes β-tagged outputs into `actors/` (e.g.
`fm_mnist_beta0.5.pth`, `Flow_loss_curve_beta0.5.png`,
`Latent_*_beta0.5.{png,npy}`), so runs with different β values never
clobber each other.

## Remote mode

A remote host connects to an already-running server (started with
`--server-mode --port <N>`) and spawns additional worker actors that pull
jobs from that server's queue:

```bash
# on the server host
./test_actor --server-mode --port 8765 --betas 0.1,0.5,1.0,2.0,4.0


## Running `latentflow.py` directly

For a single run without the actor sweep:

```bash
cd actors
../.venv/bin/python3 ./latentflow.py --beta 0.5
```

`--quiet` sends training logs to stderr and prints only the final MMD score
to stdout -- this is the mode the actor sweep uses internally to parse each
job's result.
