import math
import numpy as np


def generate_cylinder_vertices(height: float, radius: float = None, num_segments: int = 20):
    """
    This function generates the vertices of a cylinder aligned with the z-axis and centered at the origin
    and the top is at z = height

    Parameters
    ----------
    height: float
        The height of the cylinder
    radius: float
        The radius of the cylinder
    num_segments:
        The number of segments used to discretize the cylinder around the z-axis

    Returns
    -------
    vertices: list
        The vertices of the cylinder
    """

    if height is None:
        raise ValueError("Height of the cylinder must be specified")

    if radius is None:
        radius = height / 10

    # Generates the vertices of the cylinder
    vertices = []
    angle_increment = 2 * math.pi / num_segments

    for i in range(num_segments):
        x = radius * math.cos(i * angle_increment)
        y = radius * math.sin(i * angle_increment)
        vertices.append((x, y, 0))
        vertices.append((x, y, height))

    # Add top and bottom center vertices
    vertices.append((0, 0, 0))
    vertices.append((0, 0, height))

    return vertices


def displace_from_start_and_end(vertices, start: np.ndarray, end: np.ndarray) -> list:
    """
    This function displaces the vertices of a cylinder so that it matches the specified start and end points

    Parameters
    ----------
    vertices: list
        The vertices of the cylinder
    start: np.ndarray
        The start point
    end:
        The end point

    Returns
    -------
    vertices: list
        The vertices of the displaced cylinder

    """
    # Displaces the start and end points of the cylinder
    # to the specified start and end points

    # compute an homogenous matrix
    z = end - start
    z = z / np.linalg.norm(z)

    y = np.cross(z, np.array([1, 0, 0])) / np.linalg.norm(np.cross(z, np.array([1, 0, 0])))

    x = np.cross(y, z)

    homogenous_matrix = np.concatenate((x.reshape(3, 1), y.reshape(3, 1), z.reshape(3, 1), start.reshape(3, 1)), axis=1)
    homogenous_matrix = np.concatenate((homogenous_matrix, np.array([[0, 0, 0, 1]])), axis=0)

    # apply the homogenous matrix to the vertices
    for i in range(len(vertices)):
        temp = homogenous_matrix @ np.concatenate((vertices[i], np.array([1])))
        vertices[i] = tuple(temp[0:3].tolist())

    return vertices


def generate_cylinder_triangles(vertices: list) -> list:
    """
    This function generates the triangles that form the cylinder's surface

    Parameters
    ----------
    vertices: list
        List of vertices of the cylinder

    Returns
    -------
    triangles: list
        List of triangles that form the cylinder's surface

    """
    # Generates triangles from the vertices to form the cylinder's surface
    triangles = []
    num_vertices = len(vertices)

    # Generate side triangles
    for i in range(0, num_vertices - 2, 2):
        triangles.append([i, i + 1, i + 2])
        triangles.append([i + 1, i + 3, i + 2])

    # Generate bottom and top triangles
    bottom_center = num_vertices - 2
    top_center = num_vertices - 1
    for i in range(0, num_vertices - 2, 2):
        triangles.append([bottom_center, i, i + 2])
        triangles.append([top_center, i + 3, i + 1])

    # Connect last side triangle to the first one
    triangles.append([num_vertices - 2, num_vertices - 3, 0])
    triangles.append([num_vertices - 1, 1, 2])

    return triangles
