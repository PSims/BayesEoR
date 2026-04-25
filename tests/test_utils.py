import json
import importlib
from pathlib import Path
import sys
import types
from typing import Any, cast

import numpy as np
import pytest

from bayeseor import __version__


class DummyParser:
    def __init__(self) -> None:
        self.calls: list[tuple[object, Path, str, bool]] = []

    def save(
        self,
        args: object,
        path: Path,
        format: str,
        skip_none: bool,
    ) -> None:
        self.calls.append((args, path, format, skip_none))
        path.write_text(json.dumps({"saved": True}))


class DummyComm:
    def Get_rank(self) -> int:
        return 0

    def bcast(self, value: object, root: int = 0) -> object:
        return value

    def Barrier(self) -> None:
        return None


@pytest.fixture
def utils_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake_mpi4py = types.ModuleType("mpi4py")
    cast(Any, fake_mpi4py).MPI = types.SimpleNamespace(
        COMM_WORLD=DummyComm(),
        Comm=DummyComm,
    )

    monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py)
    sys.modules.pop("bayeseor.utils", None)

    return importlib.import_module("bayeseor.utils")


def test_utils_import_does_not_require_mpi4py(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("bayeseor.utils", None)
    monkeypatch.setitem(sys.modules, "mpi4py", None)

    module = importlib.import_module("bayeseor.utils")

    assert hasattr(module, "ShortTempPathManager")


@pytest.mark.parametrize(
    ("upriors_str", "nkbins", "expected"),
    [
        ("all", 4, np.array([True, True, True, True])),
        ("", 4, np.array([False, False, False, False])),
        ("1:3", 5, np.array([False, True, True, False, False])),
        ("1,3", 5, np.array([False, True, False, True, False])),
        ("-1", 4, np.array([False, False, False, True])),
    ],
)
def test_parse_uprior_inds_variants(
    utils_module: Any,
    upriors_str: str,
    nkbins: int,
    expected: np.ndarray,
) -> None:
    result = utils_module.parse_uprior_inds(upriors_str, nkbins)

    np.testing.assert_array_equal(result, expected)


def test_write_log_files_writes_version_and_args_json(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    parser = DummyParser()
    args = object()

    utils_module.write_log_files(parser, args, out_dir=tmp_path)

    assert (tmp_path / "version.txt").read_text().strip() == __version__
    assert json.loads((tmp_path / "args.json").read_text()) == {"saved": True}
    assert parser.calls == [(args, tmp_path / "args.json", "json", False)]


def test_save_and_load_numpy_dict_round_trip(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    fp = tmp_path / "payload.npy"
    arr = np.arange(4, dtype=float)
    args = {"alpha": 1}

    utils_module.save_numpy_dict(fp, arr, args, extra={"note": "ok"})
    loaded = utils_module.load_numpy_dict(fp)

    np.testing.assert_array_equal(loaded, arr)


def test_save_numpy_dict_rejects_existing_file_without_clobber(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    fp = tmp_path / "payload.npy"
    arr = np.arange(3)

    utils_module.save_numpy_dict(fp, arr, {"alpha": 1})

    with pytest.raises(ValueError, match="clobber is false"):
        utils_module.save_numpy_dict(fp, arr, {"alpha": 1})


def test_save_numpy_dict_requires_dict_extra(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    fp = tmp_path / "payload.npy"

    with pytest.raises(ValueError, match="extra must be a dictionary"):
        utils_module.save_numpy_dict(fp, np.arange(2), {"alpha": 1}, extra=["bad"])


def test_vector_is_hermitian_detects_symmetric_pairs(utils_module: Any) -> None:
    data = np.array([1 + 2j, 1 - 2j, 3 + 4j, 3 - 4j])
    conj_map = {0: 1, 1: 0, 2: 3, 3: 2}

    assert utils_module.vector_is_hermitian(data, conj_map, nt=1, nf=1, nbls=4)


def test_vector_is_hermitian_detects_asymmetry(utils_module: Any) -> None:
    data = np.array([1 + 2j, 1 - 1j])
    conj_map = {0: 1, 1: 0}

    assert not utils_module.vector_is_hermitian(data, conj_map, nt=1, nf=1, nbls=2)


def test_short_temp_path_manager_creates_and_cleans_symlink(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "real-output"
    output_dir.mkdir()
    tmp_dir = tmp_path / "links"

    manager = utils_module.ShortTempPathManager(
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        mpi_comm=DummyComm(),
    )

    assert manager.short_out_dir.is_symlink()
    assert manager.short_out_dir.resolve() == output_dir.resolve()

    manager.cleanup()

    assert not manager.short_out_dir.exists()


def test_short_temp_path_manager_requires_existing_output_dir(
    utils_module: Any,
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        utils_module.ShortTempPathManager(missing_dir, mpi_comm=DummyComm())
