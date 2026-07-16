# Prospector

# Prerequisites

- Install uv from https://docs.astral.sh/uv/getting-started/installation/
- `curl` (ships with macOS, Linux, and Windows 10+) — used to fetch large assets
- `rclone` from https://rclone.org/downloads/ — only needed if you will *upload* assets

# Running this

Large 3D assets (`.blend` / `.ply`) are **not** stored in git — they live in a
public Backblaze B2 bucket. After cloning, fetch them (no credentials needed):

```bash
./scripts/assets-pull.sh
```

See [`scripts/ASSETS.md`](scripts/ASSETS.md) for details, and for how maintainers
upload new/changed assets.

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
uv run --env-file .env -- python -m prospector.tests.test_simulation simulation.cave=chamber simulation.dynamics_model=dynamics_2d_linear
# And to render blender outputs to video
uv run --env-file .env -- python -m prospector.tests.test_blender_render_headless
uv run --env-file .env -- python -m prospector.tests.test_blender_outputs_to_video
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
