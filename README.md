## SymTensor: An Efficient Abstraction for Group-Symmetric Tensor Contractions

SymTensor aims to automate management of group symmetries in tensor contractions. With the assumption that each symmetry block is the same size, SymTensor leverages a high-order dense tensor representation of the unique (up to symmetry) tensor elements. Each contraction is performed by first contracting small tensors that represent the symmetry group, transforming the operands, and performing a single dense tensor contraction once indices are aligned. The algorithm is described in the reference at the end of this README page.

## Documentation

Complete documentation of software funcationality is generated via Sphinx and available [here](https://solomonik.cs.illinois.edu/symtensor/).

## Building and Testing

### Requirements

SymTensor relies on the [tensorbackends](https://github.com/cyclops-community/tensorbackends) package, which in turn requires NumPy and SciPy.

### Building

SymTensor can be used with Cyclops to achieve distributed parallelism and CuPy for GPU acceleration, but tensorbackends and SymTensor can be built without either Cyclops or CuPy.

The library may be installed into a Python environment from the main source directory via
```sh
pip install .
```

### Testing

Tests are contained in the subdirectory `symtensor/test/`

These may be executed as
```
python ./symtensor/test/test_nonsym.py;
python ./symtensor/test/test_pbc.py;
python ./symtensor/test/test_dmrg.py;
python ./symtensor/test/test_multi_operands.py;

## Example Codes

Aside from the test codes, a few examples of usage of symtensor with various backends are provided in `symtensor/examples`. These provide various examples for initializing tensors, including from a index-dependent function.

## Benchmarks

Benchmarks for symtensor functionality, with each of the three backends, as well as reference implementations in pure NumPy are available in `symtensor/benchmarks`.

These benchmarks are used in the performance evaluation in the paper mentioned below.

## Acknowledgements

Gao, Yang, Phillip Helms, Garnet Kin Chan, and Edgar Solomonik. "Automatic transformation of irreducible representations for efficient contraction of tensors with cyclic group symmetry." [arXiv:2007.08056](https://arxiv.org/abs/2007.08056) (2020).

This software was developed by Yang Gao, with help from Phillip Helms, Garnet Chan, and Edgar Solomonik. To acknowledge our work, we would be grateful if users cite the above paper in any publications that use SymTensor for calculations or new software.
