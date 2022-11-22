from typing import Any, Union
from pathlib import Path
import importlib.util
from casadi import MX, Function
import numpy as np


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
        return (
            Function(
                "f",
                [],
                [mx],
                [],
                ["f"],
            )
            .expand()()["f"]
            .toarray()
            .squeeze()
        )

    @staticmethod
    def to_array(value: Union[MX, np.ndarray]):
        if isinstance(value, MX):
            return TestUtils.mx_to_array(value)
        else:
            return value

    @staticmethod
    def mx_assert_equal(mx: MX, expected: Any, decimal: int = 6):
        """
        Assert that a casadi MX is equal to a numpy array if it is only numeric values
        """
        np.testing.assert_almost_equal(TestUtils.mx_to_array(mx), expected, decimal=decimal)

    @staticmethod
    def assert_equal(value: Union[MX, np.ndarray], expected: Any, decimal: int = 6):
        """
        Assert that a casadi MX or numpy array is equal to a numpy array if it is only numeric values
        """
        if isinstance(value, MX):
            TestUtils.mx_assert_equal(value, expected, decimal=decimal)
        else:
            np.testing.assert_almost_equal(value, expected, decimal=decimal)
