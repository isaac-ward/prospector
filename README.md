# Prospector

# Prerequisites

- Install uv from https://docs.astral.sh/uv/getting-started/installation/
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
# Plot cave visuals (2d or 3d) for all the caves
# provided in this repo
uv run --env-file .env -- python -m prospector.tests.test_cave_visuals
# Test cave graph (nodes and edges type graph)
# generation (automatic and manual)
uv run --env-file .env -- python -m prospector.tests.test_cave_graphs_automatic
uv run --env-file .env -- python -m prospector.tests.test_cave_graphs_manual
# Run a full simulation with the config.yaml parameters
uv run --env-file .env -- python -m prospector.tests.test_simulation

# For example, how to test with different parameters:
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=chamber
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=tunnels
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=tunnels simulation.dynamics_model=dynamics_2d_linear
```

# Citation

If you use this code in your research, please cite:

```bibtex
@misc{ward2025prospector,
  author = {Isaac Ronald Ward, Mark Paral, Kristopher Riordan, Maximilian Adang, Michelle Ho, Mykel J. Kochenderfer},
  title = {Prospector: a Cave Simulation Environment for Rotorcraft},
  year = {2025},
  publisher = {Stanford University},
  howpublished = {\url{https://github.com/isaac-ward/prospector}},
}
```
