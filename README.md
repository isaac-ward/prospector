# Prospector

# Prerequisites

- Install uv from https://uv.readthedocs.io/en/latest/installation.html
- Install Git LFS from https://git-lfs.com/

# Running this

Note that Git LFS is used to manage large files in this repository. Make sure to pull the large files after cloning the repository:

```bash
git lfs pull
```

To install dependencies use:

```bash
uv sync
```

To run the project use:

```bash
# Produces the cave graphs (i.e. nodes and edges)
# for all the caves
uv run --env-file .env -- python -m prospector.tests.test_cave_graphs
# Produces renders in matplotlib of all the caves
uv run --env-file .env -- python -m prospector.tests.test_caves_matplotlib
# Runs a simulation of the controller in the cave
# according to the config.yaml
uv run --env-file .env -- python -m prospector.tests.test_simulation

# For example, how to test with different parameters:
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=chamber
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=tunnels
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=tunnels simulation.dynamics_model=dynamics_2d_linear
```

# Citation

If you use this code in your research, please cite:

```bibtex
@misc{your_citation_key,
  author = {Isaac Ronald Ward, Mark Paral, Kristopher Riordan, Maximilian Adang, Michelle Ho},
  title = {Prospector: a Cave Simulation Environment for Rotorcraft},
  year = {2025},
  publisher = {Stanford University},
  howpublished = {\url{https://github.com/isaac-ward/prospector}},
}
```