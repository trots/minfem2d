import enum
import numpy as np
from typing import List, Optional
from scipy.sparse import lil_matrix
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Element:
    node_IDs: Optional[List] = None
    stiffness_matrix: Optional[np.ndarray] = None

    def __init__(self, node_1: int, node_2: int, node_3: int) -> None:
        self.node_IDs = [node_1, node_2, node_3]
        self.stiffness_matrix = np.array(np.zeros((3, 6)))


class ConstraintType(enum.IntFlag):
    Unknown = 0
    UX = 1
    UY = 2
    UXY = UX | UY


def int_to_constraint(value: int) -> ConstraintType:
    if value == ConstraintType.UX:
        return ConstraintType.UX

    if value == ConstraintType.UY:
        return ConstraintType.UY

    if value == ConstraintType.UXY:
        return ConstraintType.UXY

    return ConstraintType.Unknown


class Constraint:
    node_id: int = -1
    type: ConstraintType = ConstraintType.Unknown

    def __init__(self, node_id: int, type_in: ConstraintType) -> None:
        self.node_id = node_id
        self.type = type_in


def calculate_element_stiffness_matrix(element: Element, d_matrix: np.matrix, nodes_x_array: np.ndarray,
                                       nodes_y_array: np.ndarray) -> List[List[float]]:
    x = [nodes_x_array[element.node_IDs[0]], nodes_x_array[element.node_IDs[1]], nodes_x_array[element.node_IDs[2]]]
    y = [nodes_y_array[element.node_IDs[0]], nodes_y_array[element.node_IDs[1]], nodes_y_array[element.node_IDs[2]]]

    c = np.array([[1.0, 1.0, 1.0], x, y])
    c_i = np.linalg.inv(c)

    for i in range(0, 3):
        element.stiffness_matrix[0][2 * i + 0] = c_i[i][1]
        element.stiffness_matrix[0][2 * i + 1] = 0.0
        element.stiffness_matrix[1][2 * i + 0] = 0.0
        element.stiffness_matrix[1][2 * i + 1] = c_i[i][2]
        element.stiffness_matrix[2][2 * i + 0] = c_i[i][2]
        element.stiffness_matrix[2][2 * i + 1] = c_i[i][1]

    k = element.stiffness_matrix.T @ d_matrix @ element.stiffness_matrix
    c_det = np.linalg.det(c)
    k = k * abs(c_det) / 2.0
    k = np.asarray(k)

    triplets = []

    for i in range(0, 3):
        for j in range(0, 3):
            triplets.append([2 * element.node_IDs[i] + 0, 2 * element.node_IDs[j] + 0, k[2 * i + 0][2 * j + 0]])
            triplets.append([2 * element.node_IDs[i] + 0, 2 * element.node_IDs[j] + 1, k[2 * i + 0][2 * j + 1]])
            triplets.append([2 * element.node_IDs[i] + 1, 2 * element.node_IDs[j] + 0, k[2 * i + 1][2 * j + 0]])
            triplets.append([2 * element.node_IDs[i] + 1, 2 * element.node_IDs[j] + 1, k[2 * i + 1][2 * j + 1]])

    return triplets


def apply_constraints(global_stiffness_matrix: lil_matrix, constraints: List[Constraint]) -> None:
    indices_to_constraint: List[int] = []

    for constraint in constraints:
        if constraint.type & ConstraintType.UX:
            indices_to_constraint.append(2 * constraint.node_id + 0)

        if constraint.type & ConstraintType.UY:
            indices_to_constraint.append(2 * constraint.node_id + 1)

    rows, cols = global_stiffness_matrix.nonzero()

    for row, column in zip(rows, cols):
        # ((row, col), global_stiffness_matrix[row, col])
        for index_to_constraint in indices_to_constraint:
            if index_to_constraint == row or index_to_constraint == column:
                if row == column:
                    global_stiffness_matrix[row, column] = 1.0
                else:
                    global_stiffness_matrix[row, column] = 0.0


def plot_mesh_values(nodes_x_array: np.ndarray, nodes_y_array: np.ndarray, elements_array, values, title):
    triangles = []

    for element in elements_array:
        triangles.append(element.node_IDs)

    triangulation = tri.Triangulation(nodes_x_array, nodes_y_array, triangles)

    refiner = tri.UniformTriRefiner(triangulation)
    tri_refi, z_test_refi = refiner.refine_field(values, subdiv=3)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.triplot(triangulation, lw=0.5, color='white')

    c_map = cm.get_cmap(name='rainbow')
    mappable = cm.ScalarMappable(cmap=c_map)
    mappable.set_array(values)
    fig.colorbar(mappable, ax=ax)
    ax.tricontourf(tri_refi, z_test_refi, cmap=c_map)
    ax.tricontour(tri_refi, z_test_refi)

    ax.set_title(title)

    plt.show()
