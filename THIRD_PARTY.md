# Data and dependency notices

The example fixture is generated entirely by `player_value/fixture.py` using Python's pseudorandom generator. It contains invented player IDs, random lineups, and sampled outcomes. No external dataset, athlete likeness, logo, screenshot, trained checkpoint, or proprietary rating table is included. The historical evidence files contain aggregate counts and evaluation summaries; they do not grant rights to any underlying sports data.

Dependencies are installed from their distributors and are not vendored in this repository. Their license notices remain in the installed distributions:

| Dependency | License / authoritative source |
|---|---|
| PyTorch | [BSD-style license and incorporated component notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| NumPy | [BSD-3-Clause and bundled-component notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SymPy | [BSD license](https://github.com/sympy/sympy/blob/master/LICENSE) |
| NetworkX | [BSD-3-Clause](https://github.com/networkx/networkx/blob/main/LICENSE.txt) |
| Jinja2 / MarkupSafe | [Jinja license](https://github.com/pallets/jinja/blob/main/LICENSE.txt), [MarkupSafe license](https://github.com/pallets/markupsafe/blob/main/LICENSE.txt) |
| fsspec | [BSD-3-Clause](https://github.com/fsspec/filesystem_spec/blob/master/LICENSE) |
| filelock | [Unlicense](https://github.com/tox-dev/filelock/blob/main/LICENSE) |
| typing-extensions | [PSF license](https://github.com/python/typing_extensions/blob/main/LICENSE) |
| mpmath | [BSD license](https://github.com/mpmath/mpmath/blob/master/LICENSE) |
| setuptools | [MIT license](https://github.com/pypa/setuptools/blob/main/LICENSE) |

Methodological references: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) for logits-based classification losses; [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) for the limits of cross-platform reproducibility; and [Dunks & Threes' EPM explanation](https://dunksandthrees.com/about/epm) for the historical external comparator. EPM data is not required for the public example.
