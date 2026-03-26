import importlib
import sys


def test_model_package_uses_lazy_imports_for_submodules():
    for module_name in [
        "bayeseor.model",
        "bayeseor.model.healpix",
        "bayeseor.model.instrument",
        "bayeseor.model.k_cube",
        "bayeseor.model.noise",
    ]:
        sys.modules.pop(module_name, None)

    model = importlib.import_module("bayeseor.model")

    assert "bayeseor.model.healpix" not in sys.modules
    assert "bayeseor.model.instrument" not in sys.modules
    assert "bayeseor.model.k_cube" not in sys.modules
    assert "bayeseor.model.noise" not in sys.modules

    healpix_class = model.Healpix

    assert healpix_class.__name__ == "Healpix"
    assert "bayeseor.model.healpix" in sys.modules
    assert "bayeseor.model.instrument" not in sys.modules
    assert "bayeseor.model.k_cube" not in sys.modules
    assert "bayeseor.model.noise" not in sys.modules
