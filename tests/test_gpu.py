import ctypes
import sys
import types
from pathlib import Path
from typing import Any, cast

from bayeseor.gpu import GPUInterface


class FakeFunction:
    def __init__(self) -> None:
        self.argtypes: list[object] | None = None


class FakeMagmaLibrary:
    def __init__(self) -> None:
        self.magma_init = FakeFunction()
        self.magma_finalize = FakeFunction()
        self.magma_zpotrf = FakeFunction()


class FakeDevice:
    def __init__(self, index: int) -> None:
        self.index = index

    @staticmethod
    def count() -> int:
        return 2

    def name(self) -> str:
        return f"Fake GPU {self.index}"


def test_gpu_interface_falls_back_when_pycuda_is_unavailable(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    sys.modules.pop("pycuda", None)
    sys.modules.pop("pycuda.autoinit", None)
    sys.modules.pop("pycuda.driver", None)

    interface = GPUInterface(verbose=False)

    assert not interface.gpu_initialized
    captured = capsys.readouterr()
    assert "Exception loading GPU encountered" in captured.out


def test_gpu_interface_initializes_with_stubbed_cuda_dependencies(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    fake_pycuda = types.ModuleType("pycuda")
    fake_autoinit = types.ModuleType("pycuda.autoinit")
    fake_driver = types.ModuleType("pycuda.driver")
    cast(Any, fake_driver).Device = FakeDevice
    cast(Any, fake_pycuda).autoinit = fake_autoinit
    cast(Any, fake_pycuda).driver = fake_driver

    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.autoinit", fake_autoinit)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_driver)
    monkeypatch.setattr(ctypes, "CDLL", lambda _: FakeMagmaLibrary())

    interface = GPUInterface(base_dir=tmp_path, rank=0, verbose=True)

    assert interface.gpu_initialized
    assert interface.base_dir == tmp_path
    assert interface.magma_zpotrf.argtypes is not None

    captured = capsys.readouterr()
    assert "Loading shared library" in captured.out
    assert "Computing on GPU(s)" in captured.out
    assert "Rank 0: 2 GPUs (Fake GPU 0, Fake GPU 1)" in captured.out
