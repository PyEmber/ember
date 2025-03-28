# Contributing to Docs

## Getting Started 

Assumes you have:
* A local clone of Ember.
* Python >= 3.12 (Materials for MkDocs requirement).

Within your choice of virtual environment,
```
pip install mkdocs-material
```

Render a local view of the docs (you will need this to PR content later),
```
# From the ember source directory,
cd docs/
mkdocs serve
```

To add/edit content, find the corresponding files under `ember/docs/docs`.

To make asthetic changes, edit `mkdocs.yml`.

See the official Materials for MkDocs [docs](https://squidfunk.github.io/mkdocs-material) for further guidance.

## References
* https://squidfunk.github.io/mkdocs-material 