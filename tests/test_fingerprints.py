import numpy as np
import pytest


def test_fp_shape():
    from src.features.fingerprints import morgan_fp
    fp = morgan_fp("CCO")
    assert fp.shape == (2048,)


def test_fp_zeros_on_invalid():
    from src.features.fingerprints import morgan_fp
    fp = morgan_fp("NOT_A_MOLECULE$$")
    assert fp.sum() == 0


def test_fp_different_mols():
    from src.features.fingerprints import morgan_fp
    fp1 = morgan_fp("CCO")
    fp2 = morgan_fp("c1ccccc1")
    assert not np.array_equal(fp1, fp2)


def test_descriptors_shape():
    from src.features.fingerprints import rdkit_descriptors
    desc = rdkit_descriptors("CCO", n=50)
    assert desc.shape == (50,)
