# kde-learn

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://kde-learn.readthedocs.io/en/latest//"><img alt="Docs: Sphinx" src="https://readthedocs.org/projects/pip/badge/?version=latest&style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## Installation

```shell
conda create -n kde-learn python=3.9.12
conda activate kde-learn

pip install -e .
```

## Code Prettifiers

```
black {source_file_or_directory}
flake8 {source_file_or_directory}
isort {source_file_or_directory}
mypy {source_file_or_directory} - this one have to be executed manually (it is not included in CI and pre-commit)
```

They are also executed as a pre-commit hooks. See more in `.pre-commit-config.yaml`.

Useful links:
 - [Using black with other tools](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html)

## Unit tests

Basic pytest usage:
```
pytest tests/
```

with coverage
```
pytest tests/ --cov=kdelearn/ --cov-report term-missing
```

## Sphinx

Useful links:
 - [style guide](https://sphinx-rtd-theme.readthedocs.io/en/stable/demo/lists_tables.html)
 - [style guide2](https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html)


### Create the documentation

Docstrings inspired on scikit-learn.

```
cd docs/
make clean
make html
```

### Publish on readthedocs.org

1. Log in with GitHub credentials
2. Import a project
3. Build the project

## Convert notebook into pdf
```
jupyter nbconvert --to pdf --no-input example_notebook.ipynb
```
