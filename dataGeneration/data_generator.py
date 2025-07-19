import pandas as pd
import numpy as np
import os

import re

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from random import randrange
import pickle
import csv

import sympy
from sympy import symbols, lambdify
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from multiprocessing import Pool

vocab = {'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 'sin(x)': 11,'cos(x)': 12, 'sin(y)': 13, 'cos(y)': 14, 'sin(z)': 15, 'cos(z)': 16, '-': 17, '+': 18, '*': 19, '.': 20 }

first_order_term = ['cos(x)', 'cos(y)', 'cos(z)', 'sin(x)', 'sin(y)', 'sin(z)', 'cos(2*x)', 'cos(2*y)', 'cos(2*z)', 'sin(2*x)', 'sin(2*y)', 'sin(2*z)']
second_order_term = ['cos(x)*cos(y)', 'cos(x)*sin(y)', 'cos(x)*cos(z)', 'cos(x)*sin(z)', 'cos(y)*cos(z)', 'cos(y)*sin(z)', 'sin(x)*cos(y)', 'sin(x)*sin(y)', 'sin(y)*sin(z)', 'sin(y)*cos(z)', 'sin(x)*cos(z)', 'sin(x)*sin(z)',\
                    'cos(x)*cos(x)', 'cos(y)*cos(y)', 'cos(z)*cos(z)', 'sin(x)*sin(x)', 'sin(y)*sin(y)', 'sin(z)*sin(z)']
third_order_term = ['cos(x)*cos(y)*cos(z)','sin(x)*sin(y)*sin(z)']

term_list = first_order_term + second_order_term + third_order_term

count_term = 0
count_eq = 0
max_integer = 5
max_decimal = 10
min_terms = 3
max_terms = 3
levelset = 0

def generate_coefficient(max_integer, max_decimal):

    coefficient_integer = randrange(0,max_integer+1)
    if coefficient_integer == 0:
        coefficient_decimal = randrange(1,max_decimal)
    else:
        coefficient_decimal = randrange(max_decimal)
    coefficient = f"{coefficient_integer}.{coefficient_decimal}"

    return coefficient

def is_equation_dependent_on_xyz(equation):
    """
    Check if the given equation depends on all three variables x, y, and z.
    
    Parameters:
    - equation (str): The equation string to check.
    
    Returns:
    - bool: True if the equation depends on x, y, and z; False otherwise.
    """
    # Check for the presence of each variable in the equation string
    depends_on_x = 'x' in equation
    depends_on_y = 'y' in equation
    depends_on_z = 'z' in equation
    
    # The equation is dependent on all three variables if all are present
    return depends_on_x and depends_on_y and depends_on_z

def find_connected_components(verts, faces):
    """
    Find connected components in mesh data.
    
    Parameters:
    - verts: Vertices of the mesh (N x 3 array for N vertices).
    - faces: Faces of the mesh (M x 3 array for M triangular faces).
    
    Returns:
    - labels: Array of component labels for each vertex.
    - num_components: Number of connected components found.
    """
    # Create an adjacency matrix for the vertices
    num_verts = verts.shape[0]
    rows = np.hstack([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.hstack([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones_like(rows)
    adj_matrix_sparse = csr_matrix((data, (rows, cols)), shape=(num_verts, num_verts))
    num_components, labels = connected_components(csgraph=adj_matrix_sparse, directed=False)
    return labels, num_components

def get_actual_verts(verts, min_val, max_val, res_check):
    return min_val + verts * (max_val - min_val) / (res_check - 1)

def is_within_bounds(verts, t):        
    return np.all(verts >= 0) and np.all(verts <= t) and np.max(verts, axis = 0).min() == t and np.min(verts, axis = 0).max() == 0.

def is_mesh_valid(equation):
    levelset = 0.
    x, y, z = symbols('x y z')

    # Scaling the variables for periodicity of 2
    scaled_equation = equation.replace('x', '(2*pi/20) * x').replace('y', '(2*pi/20) * y').replace('z', '(2*pi/20) * z')
    
    f_expr = sympy.sympify(scaled_equation)  # Convert string to SymPy expression
    f = lambdify((x, y, z), f_expr, 'numpy')  # Convert symbolic expr to a numpy-callable function

    t = 20  # Domain size [0, 20]
    res_check = 20  # Resolution
    step = res_check * 1j

    # Generate the grid within [0, 20]
    x = np.linspace(0, t, res_check)  # t = 20
    y = np.linspace(0, t, res_check)
    z = np.linspace(0, t, res_check)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    V = f(X, Y, Z)

    # Check if the isosurface crosses the levelset
    if V.min() < levelset < V.max():
        vertices, faces, _, _ = marching_cubes(V, level=levelset)

        # Rescale the vertices to match the physical domain
        verts_x = get_actual_verts(vertices[:, 0], X.min(), X.max(), res_check)
        verts_y = get_actual_verts(vertices[:, 1], Y.min(), Y.max(), res_check)
        verts_z = get_actual_verts(vertices[:, 2], Z.min(), Z.max(), res_check)

        verts = np.column_stack((verts_x, verts_y, verts_z))

        if is_within_bounds(verts, t):
            labels, num_components = find_connected_components(verts, faces)
            if num_components == 1:
                return True

    return False

def generate_valid_implicit_equations(_):
    """
    Generates a single equation and checks its validity.
    This function is intended to be run in a multiprocessing pool.
    """
    while True:
        eq = ''
        num_term = randrange(min_terms, max_terms + 1)
        # if num_term == 1:
        #     eq_terms = np.random.choice(third_order_term, 1)
        # else:
        eq_terms = np.random.choice(term_list, num_term, replace = False).tolist()

        for term in eq_terms:
            coefficient = generate_coefficient(max_integer, max_decimal)

            sign = np.random.choice(['-', '+'])

            eq += f"{sign}{coefficient}*{term}"
        const = generate_coefficient(max_integer, max_decimal)
        const_sign = np.random.choice(['-', '+'])
        eq += f"{const_sign}{const}"
        eq = eq.lstrip('+')
        
        is_dependent = is_equation_dependent_on_xyz(eq)
        is_valid = is_mesh_valid(eq)
        
        if is_dependent and is_valid:
            return eq

# eq_example = '3.7*cos(y) - 1.8*cos(z) + 5.3*cos(x)*sin(z) - 1.7'

eq_list = []

def main(num_eqs, num_processes):
    """
    Main function to generate equations using multiprocessing.
    """
    worker_func = generate_valid_implicit_equations

    with Pool(processes=num_processes) as pool:
        results = pool.map(worker_func, range(num_eqs))

    return results

if __name__ == '__main__':

    num_eqs = 20000  # Number of equations to generate
    num_processes = 20  # Number of parallel processes

    valid_output_filename = '../data/implicit_equations/valid_tpms_equations.csv'

    start = time.time()
    results = main(num_eqs, num_processes)

    valid_eqs = results
    invalid_eqs = []
    end = time.time()

    print(f"Generated {num_eqs} TPMS Equations")
    print(f"     #valid eqs = {len(valid_eqs)}")
    print(f"     #invalid eqs = {len(invalid_eqs)}")
    print(f"Time = {end-start: .4f} s")

    with open(valid_output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(['equation'])

        for eq in valid_eqs:
            writer.writerow([eq])