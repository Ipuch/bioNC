from typing import Any
from pathlib import Path
import importlib.util
from casadi import MX, Function


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

    @staticmethod
    def to_casadi_func(mx: MX):
        return Function(
            "f",
            [],
            [mx],
            [],
            ["f"],
        ).expand()

    @staticmethod
    def mx_to_array(mx: MX):
        """
        Convert a casadi MX to a numpy array if it is only numeric values
        """
        return Function(
            "f",
            [],
            [mx],
            [],
            ["f"],
        ).expand()()["f"].toarray().squeeze()
