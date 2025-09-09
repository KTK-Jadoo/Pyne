# Confident Learning — Auto Auditor (Gradio App)

Fast, interactive auditing for mislabeled data using the Confident Learning approach, wrapped in a simple Gradio UI.

Note: This repository is being prepared for GitHub. If you previously used this as a Hugging Face Space, see the “Hugging Face Spaces” section below to restore Space metadata.

## Features

- Label-quality audit: surfacing likely label errors using Confident Learning.
- Interactive review: sort/filter suspected issues and accept/reject fixes.
- Exportable results: download reports and optionally a cleaned dataset.
- Simple UI: powered by Gradio for quick local or hosted use.

## Quickstart

Prerequisites:

- Python 3.10+

Set up and run locally:

```bash
git clone <YOUR_GITHUB_REPO_URL>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate  # or use uv/pipenv/conda
pip install -r requirements.txt                     # TODO: add this file
python app.py                                       # TODO: confirm entrypoint
```

Then open the Gradio URL shown in the terminal (typically http://127.0.0.1:7860).

## Usage

1. Launch the app (see Quickstart).
2. Upload your dataset (e.g., CSV). TODO: document accepted formats.
3. Select the target/label column. TODO: document feature columns and dtypes.
4. Run the audit to compute label-error likelihoods.
5. Review flagged rows and export a report or a corrected dataset.

## Configuration

Add configurable options as needed (examples):

- Model/embeddings backend (e.g., scikit-learn baseline, transformer embeddings).
- Audit thresholds and sampling strategies.
- Columns and preprocessing options for text/tabular inputs.

Document these in a `config.toml` or environment variables as the app evolves.

## Project Structure

This README assumes a typical Gradio app layout. Adjust to match your code.

```
.
├─ app.py                 # Gradio UI + inference/auditing hooks (TODO)
├─ requirements.txt       # Python dependencies (TODO)
├─ src/                   # Core logic (optional)
├─ data/                  # Sample datasets (optional)
└─ README.md
```

## Development

- Create an isolated environment (venv/conda/uv) and install dev deps.
- Add type checks (mypy/pyright) and linting (ruff/flake8) if desired.
- Write unit tests for the auditing core before UI integrations.

Common commands (examples):

```bash
pip install -r requirements-dev.txt  # TODO: optional dev deps
pytest -q                            # run tests
python app.py                        # run the UI
```

## Hugging Face Spaces

If you deploy this project as a Hugging Face Space, Spaces read metadata from the top of `README.md` in YAML frontmatter. Here is the metadata that used to live here — paste it back at the top of this file when pushing to Spaces to restore your Space settings:

```yaml
---
title: Confident Learning
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
short_description: Confidence Learning Auto Auditor
---
```

Reference: https://huggingface.co/docs/hub/spaces-config-reference

## Roadmap

- [ ] Publish `requirements.txt` and confirm `app.py` entrypoint
- [ ] Add sample dataset and screenshots/GIF
- [ ] Expose configuration via UI and `config.toml`
- [ ] Add tests for the auditing core
- [ ] Optional: provide a CLI for batch audits

## Contributing

Contributions are welcome! Please open an issue to discuss major changes.

Recommended steps:

- Fork the repo and create a feature branch.
- Keep changes focused; add tests where practical.
- Open a PR with a clear description and before/after context.

## Acknowledgements

- Confident Learning (Northcutt et al.) and the open-source ecosystem around label-error detection.
- Gradio for fast UI prototyping.
