from typing import Any
from pathlib import Path
import importlib.util


class TestUtils:
    @staticmethod
    def bionc_folder() -> str:
        return str(Path(__file__).parent / "..")

    @staticmethod
    def load_module(path: str) -> Any:
        module_name = path.split("/")[-1].split(".")[0]
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
