import numpy as np
from pathlib import Path

import pytest

from bayeseor.matrices import BuildMatrices

@pytest.fixture(scope="function")
def build_matrices_params():
    return dict()


@pytest.fixture(scope="function")
def build_matrices(build_matrices_params):
    BM = BuildMatrices(

    )