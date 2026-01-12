"""
Mixin classes providing __repr__ and __str__ methods for natural coordinates, velocities, and accelerations.

These mixins can be inherited by both numpy and casadi implementations to avoid code duplication.
"""


class SegmentNaturalCoordinatesReprMixin:
    """Mixin providing string representations for SegmentNaturalCoordinates."""

    def __repr__(self) -> str:
        return f"SegmentNaturalCoordinates(u={self.u}, rp={self.rp}, rd={self.rd}, w={self.w})"

    def __str__(self) -> str:
        out = "SegmentNaturalCoordinates:\n"
        out += f"  u  = {self.u}\n"
        out += f"  rp = {self.rp}\n"
        out += f"  rd = {self.rd}\n"
        out += f"  w  = {self.w}\n"
        return out


class NaturalCoordinatesReprMixin:
    """Mixin providing string representations for NaturalCoordinates."""

    def __repr__(self) -> str:
        return f"NaturalCoordinates(nb_segments={self.nb_qi()}, shape={self.shape})"

    def __str__(self) -> str:
        out = f"NaturalCoordinates with {self.nb_qi()} segment(s)\n"
        for i in range(self.nb_qi()):
            out += f"  Segment {i}: u={self.u(i)}, rp={self.rp(i)}, rd={self.rd(i)}, w={self.w(i)}\n"
        return out


class SegmentNaturalVelocitiesReprMixin:
    """Mixin providing string representations for SegmentNaturalVelocities."""

    def __repr__(self) -> str:
        return f"SegmentNaturalVelocities(udot={self.udot}, rpdot={self.rpdot}, rddot={self.rddot}, wdot={self.wdot})"

    def __str__(self) -> str:
        out = "SegmentNaturalVelocities:\n"
        out += f"  udot  = {self.udot}\n"
        out += f"  rpdot = {self.rpdot}\n"
        out += f"  rddot = {self.rddot}\n"
        out += f"  wdot  = {self.wdot}\n"
        return out


class NaturalVelocitiesReprMixin:
    """Mixin providing string representations for NaturalVelocities."""

    def __repr__(self) -> str:
        return f"NaturalVelocities(nb_segments={self.nb_qdoti()}, shape={self.shape})"

    def __str__(self) -> str:
        out = f"NaturalVelocities with {self.nb_qdoti()} segment(s)\n"
        for i in range(self.nb_qdoti()):
            out += f"  Segment {i}: udot={self.udot(i)}, vdot={self.vdot(i)}, wdot={self.wdot(i)}\n"
        return out


class SegmentNaturalAccelerationsReprMixin:
    """Mixin providing string representations for SegmentNaturalAccelerations."""

    def __repr__(self) -> str:
        return f"SegmentNaturalAccelerations(uddot={self.uddot}, rpddot={self.rpddot}, rdddot={self.rdddot}, wddot={self.wddot})"

    def __str__(self) -> str:
        out = "SegmentNaturalAccelerations:\n"
        out += f"  uddot  = {self.uddot}\n"
        out += f"  rpddot = {self.rpddot}\n"
        out += f"  rdddot = {self.rdddot}\n"
        out += f"  wddot  = {self.wddot}\n"
        return out


class NaturalAccelerationsReprMixin:
    """Mixin providing string representations for NaturalAccelerations."""

    def __repr__(self) -> str:
        return f"NaturalAccelerations(nb_segments={self.nb_qddoti()}, shape={self.shape})"

    def __str__(self) -> str:
        out = f"NaturalAccelerations with {self.nb_qddoti()} segment(s)\n"
        for i in range(self.nb_qddoti()):
            out += f"  Segment {i}: uddot={self.uddot(i)}, vddot={self.vddot(i)}, wddot={self.wddot(i)}\n"
        return out
