from casadi import Function, MX


def to_numeric_MX(mx: MX, expand: bool = False) -> MX:
    """
    This function is intended to be use to convert a casadi MX to a numeric casadi MX.
    It useful when some of the operators are not supported by the MX class to be expanded

    Parameters
    ----------
    mx: MX
        The casadi MX to convert
    expand: bool
        If True, the function will be expanded

    """
    if expand:
        return MX(
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
    else:
        return MX(
            Function(
                "f",
                [],
                [mx],
                [],
                ["f"],
            )()["f"]
            .toarray()
            .squeeze()
        )
