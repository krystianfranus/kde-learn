# kde-learn

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://kde-learn.readthedocs.io/en/latest//"><img alt="Docs: Sphinx" src="https://readthedocs.org/projects/pip/badge/?version=latest&style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

## Installation

```shell
conda create -n kde-learn python=3.8.12
conda activate kde-learn

pip install -e .
```

## Helpers

```
black {source_file_or_directory}
flake8 {source_file_or_directory}
isort {source_file_or_directory}
```

Pre-commit works with 'git commit'

## Sphinx

### Create the documentation

```
cd docs/
make html
```

### Publish on readthedocs.org

1. Log in with GitHub credentials
2. Import a project
3. Build the project
