import numpy as np
import rerun as rr


def display_frame(
        animation_id, scale: list[float] = 1, timeless: str = True, homogenous_transform: np.ndarray = None,
        colors: list[tuple] = None,
        axes: list[str] = None,
) -> None:
    """Display the world reference frame"""

    origin = homogenous_transform[:3, 3] if homogenous_transform is not None else np.zeros(3)
    vectors = homogenous_transform[:3, :3] if homogenous_transform is not None else np.eye(3)
    scale = scale if isinstance(scale, list) else [scale] * 3
    colors = colors if colors is not None else [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    axes = axes if axes is not None else ["X", "Y", "Z"]

    for i, (axis, color) in enumerate(zip(axes, colors)):
        rr.log(
            f"{animation_id}/{axis}",
            rr.Arrows3D(
                origins=origin,
                vectors=vectors[:, i] * scale[i],
                colors=np.array(color),
            ),
            timeless=timeless,
        )
