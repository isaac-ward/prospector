# Template Python Repository

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
uv run python src/prospector/main.py
uv run --env-file .env -- python -m prospector.tests.test_caves_matplotlib
```

# Citation

If you use this code in your research, please cite:

```bibtex
@misc{your_citation_key,
  author = {Isaac Ronald Ward, Mark Paral, Kristopher Riordan, Maximilian Adang, Michelle Ho},
  title = {Prospector: a Cave Simulation Environment for Training Field Robots},
  year = {2025},
  publisher = {Stanford University},
  howpublished = {\url{https://github.com/isaac-ward/prospector}},
}
```