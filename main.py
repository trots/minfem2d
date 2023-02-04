import sys
import math
from typing import List, Optional
from scipy.sparse import lil_matrix
from scipy.sparse import linalg
import numpy as np
import meshio
import minfem2d as mf

nodes_x_array: Optional[np.ndarray] = None
nodes_y_array: Optional[np.ndarray] = None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Not enough arguments")
        exit(0)

    print("File In: ", sys.argv[1])

    input_file = open(sys.argv[1], "r")
    input_file_lines: List[str] = input_file.readlines()

    # Load coefficients

    line: int = 0
    words: List[str] = input_file_lines[line].split()
    poisson_coefficient: float = float(words[0])
    young_module: float = float(words[1])
    print("Poisson coefficient: ", poisson_coefficient)
    print("Young module: ", young_module)

    # Create stiffness matrix

    stiffness_matrix: np.matrix = np.matrix([[1.0, poisson_coefficient, 0.0],
                                             [poisson_coefficient, 1.0, 0.0],
                                             [0.0, 0.0, (1.0 - poisson_coefficient) / 2.0]])
    coefficient: float = young_module / (1.0 - (poisson_coefficient * poisson_coefficient))
    stiffness_matrix *= coefficient
    print("Stiffness matrix:\n", stiffness_matrix)

    # Load nodes of the FE mesh

    line += 1
    words = input_file_lines[line].split()
    nodes_count = int(words[0])
    nodes_x_array = np.zeros(nodes_count)
    nodes_y_array = np.zeros(nodes_count)

    for i in range(nodes_count):
        line += 1
        words = input_file_lines[line].split()
        nodes_x_array[i] = float(words[0])
        nodes_y_array[i] = float(words[1])

    print("Nodes count:", nodes_count)

    # Load elements of the FE mesh

    line += 1
    words = input_file_lines[line].split()
    elements_count: int = int(words[0])
    elements_array: List[mf.Element] = []
    for i in range(elements_count):
        line += 1
        words = input_file_lines[line].split()
        elements_array.append(mf.Element(int(words[0]), int(words[1]), int(words[2])))
    print("Elements count:", len(elements_array))

    # Load constraints

    line += 1
    words = input_file_lines[line].split()
    constraints_count: int = int(words[0])
    constraints_array: List[mf.Constraint] = []
    for i in range(constraints_count):
        line += 1
        words = input_file_lines[line].split()
        constraints_array.append(mf.Constraint(int(words[0]), mf.int_to_constraint(int(words[1]))))
    print("Constraints count:", len(constraints_array))

    # Load loads

    line += 1
    words = input_file_lines[line].split()
    loads_count: int = int(words[0])
    loads_array: np.ndarray = np.zeros((2 * nodes_count, 1))
    for i in range(loads_count):
        line += 1
        words = input_file_lines[line].split()
        node: int = int(words[0])
        x: float = float(words[1])
        y: float = float(words[2])
        loads_array[2 * node + 0] = x
        loads_array[2 * node + 1] = y
    print("Loads count:", loads_count)

    input_file.close()

    # Calculate global stiffness matrix

    print("Calculate global stiffness matrix...", end=" ")

    triplets: List[List[float]] = []

    for element in elements_array:
        e_triplets = mf.calculate_element_stiffness_matrix(element, stiffness_matrix, nodes_x_array, nodes_y_array)

        for e_triplet in e_triplets:
            triplets.append(e_triplet)

    global_stiffness_matrix = lil_matrix((2 * nodes_count, 2 * nodes_count))

    for triplet in triplets:
        global_stiffness_matrix[triplet[0], triplet[1]] = global_stiffness_matrix[triplet[0], triplet[1]] + triplet[2]

    print("Ok")
    print("Global stiffness matrix shape:", global_stiffness_matrix.shape)

    # Apply constraints

    print("Apply constraints...", end=" ")
    mf.apply_constraints(global_stiffness_matrix, constraints_array)
    print("Ok")

    # Solve

    print("Solve...", end=" ")
    displacements = linalg.spsolve(global_stiffness_matrix.tocsr(), loads_array)
    print("Ok")

    # Mises

    print("Von Mises stress calculation...", end=" ")

    sigma_mises_array = []

    for element in elements_array:
        delta = np.array([displacements[2 * element.node_IDs[0] : 2 * element.node_IDs[0] + 2],
                          displacements[2 * element.node_IDs[1] : 2 * element.node_IDs[1] + 2],
                          displacements[2 * element.node_IDs[2] : 2 * element.node_IDs[2] + 2]])
        delta = delta.reshape(6)
        sigma = stiffness_matrix @ element.stiffness_matrix @ delta
        sigma = np.array(sigma.tolist()[0])
        sigma_mises = math.sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] +
                                3.0 * sigma[2] * sigma[2])
        sigma_mises_array.append(sigma_mises)

    print("Ok")

    # Draw results

    result_file_name = input_file.name + "_results.vtk"
    print("Write results to", result_file_name + "...", end=" ")

    points = np.zeros((nodes_count, 3))

    for i in range(0, nodes_count):
        points[i][0] = nodes_x_array[i]
        points[i][1] = nodes_y_array[i]

    triangles = []

    for element in elements_array:
        triangles.append(element.node_IDs)

    cells = {
        "triangle": np.array(triangles)
    }

    meshio.write_points_cells(result_file_name, points, cells,
                              cell_data={'cell_data': {'von_mises_stress': np.array(sigma_mises_array)}},
                              point_data={'displacement_x': displacements[::2], 'displacement_y': displacements[1::2]})

    print("Ok")
    print("Complete")
