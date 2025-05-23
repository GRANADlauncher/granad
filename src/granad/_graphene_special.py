
from matplotlib.path import Path
import jax.numpy as jnp
import jax
from typing import Literal, Callable
import matplotlib.path as mpltPath
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
def get_polygon(n: Literal[3,6], side_length: float, orientation_angle: float, min_side_length: float, side_length_increment: float, shift_vec_for_edge_alignment: Callable[[int],jax.Array]) -> jax.Array:
    """
    Creates a regular triangle or hexagon of suitable side length (<= given side length) containing maximum number of hexagon (from lattice) per side.
    Then it put the shape on top of the big material sheet and orient the polygon as per edge type requriment.
    
    Parameters:
        n (int): Order of the polygon. Only 3 and 6 are allowed.
        side_length (float): Side length of the polygon,the user wants to build. (Angstrom) 
        orientation_angle: Rotates the polygon during creation.
        min_side_length (float): Minimum possible side length of the polygon depending on the shape (triangle or hexagon) and edge type of the flake to be cut.
        side_length_increment (float): Quantum of side length increment if one more hexagon (from lattice) is to be included in the flake.
        shift_vec_for_edge_alignment (Callable[[int],jax.Array]): Depending on odd or even number of hexagon present per side,
        this function returns a vector to orient the polygon in specific direction to cut specific type of edge. 
    Returns:
        (jax.Array): Coordinates of the properly oriented polygon.
        
    """
    ##  Side length increases as integer multiple of side_length_increment ==> 0,1,2,3...
    k=0 if side_length<=min_side_length else (side_length-min_side_length)//side_length_increment
    side_length=min_side_length+k*side_length_increment
    radius=side_length/(2*jnp.cos(jnp.pi/2*(1-2/n)))
    phi=orientation_angle
    R=jnp.array([[jnp.cos(phi),-jnp.sin(phi),0],
                [jnp.sin(phi),jnp.cos(phi),0],
                [0,0,1]])
    polygon=jax.vmap(lambda theta : jnp.array([radius*jnp.sin(theta+phi),radius*jnp.cos(theta+phi),0]))(jnp.linspace(0,2*jnp.pi,n+1)[:-1])
    #The created polygon may not be on top of the sheet of materialmin_x_polygon, min_y_polygon=jnp.min(polygon[:,0]), jnp.min(polygon[:,1])
    #We assume the the bottom-left lattice point of the rectangular material sheet is kept at origin (0,0,0)
    min_x_polygon, min_y_polygon=jnp.min(polygon[:,0]), jnp.min(polygon[:,1])
    #Shift the polygon such that it alings perfectly with the sheet
    polygon_sheet_alignment_shift_vec=jnp.array([min_x_polygon,min_y_polygon,0])
    aligned_polygon=polygon-polygon_sheet_alignment_shift_vec
    #depending on #hexagon present per side, align the polygon properly to cut out desired edge type
    return aligned_polygon+shift_vec_for_edge_alignment(k) 

# Define a rectangular graphene sheet
def get_graphene_sheet(m,n,cc_dist):
    def translation_vector(m,n):
        return jnp.array([3*cc_dist*m,jnp.sqrt(3)*cc_dist*n,0])
    
    unit_cell=jnp.array([[0,0,0],
          [cc_dist/2,jnp.sqrt(3)/2*cc_dist,0],
          [3/2*cc_dist,jnp.sqrt(3)/2*cc_dist,0],
          [2*cc_dist,0,0]])
    m_grid,n_grid=jnp.meshgrid(jnp.arange(1,m+1),jnp.arange(1,n+1))
    flatten_idx_grid= jnp.column_stack((m_grid.ravel(), n_grid.ravel()))
    shift=jnp.array([0,0,0])
    coords=jax.vmap(lambda m,n: shift+unit_cell+translation_vector(m,n),in_axes=(0,0))(flatten_idx_grid[:,0],flatten_idx_grid[:,1]).reshape(m*n*unit_cell.shape[0],3)
    return coords-coords[0] # To make the bottom-left lattice coordinate (0,0,0)

#assign sublattice label to a given flake
def assign_sublattice(coords,cc_dist):
    sublattice_labels=[None]*len(coords)
    
    nodes_to_find_neighbour=[0]
    sublattice_labels[0]='A'
    
    kdtree = KDTree(coords)
    
    while nodes_to_find_neighbour:
        current_node=nodes_to_find_neighbour.pop(0)
        neighbours=kdtree.query_ball_point(coords[current_node], 1.1*cc_dist)
        for neighbour in neighbours:
            if sublattice_labels[neighbour]== None:
                sublattice_labels[neighbour]= 'B' if sublattice_labels[current_node]=='A' else 'A'
                nodes_to_find_neighbour.append(neighbour)
            
    return sublattice_labels

def remove_single_neighbor_points(coords, radius=1.0):
    """
    Removes points in a set of lattice coordinates that have only one neighbor.

    Args:
        coords: A NumPy array of coordinates, where each row represents a point.
        radius: The distance threshold for considering neighbors (default is 1.0).

    Returns:
        A NumPy array containing the coordinates of points with more than one neighbor.
    """

    tree = KDTree(coords)
    points_to_keep = []

    for i, point in enumerate(coords):
        # Find all neighbors within the given radius
        neighbors = tree.query_ball_point(point, r=radius)
        # Count the number of neighbors (excluding the point itself)
        if len(neighbors) > 2:  # More than one neighbor
            points_to_keep.append(point)

    return jnp.array(points_to_keep)

def _cut_flake_graphene(polygon_id, edge_type, side_length, lattice_constant):
    # measure everything in units of the lc
    a = lattice_constant/jnp.sqrt(3)
    mapping = {
    "triangle_armchair" : [3, side_length, 0, 6*a, 3*a, lambda k: jnp.array([1,0,0])*a],
    "triangle_zigzag" : [3, side_length, jnp.pi/2, 3*jnp.sqrt(3)*a, jnp.sqrt(3)*a, lambda k: jnp.array([0,0,0])*a],
    "hexagon_armchair" : [6, side_length, jnp.pi/2, 4*a, 3*a, lambda k: jnp.array([0,jnp.sqrt(3)/4*(1-(-1)**k)*a,0])],
    "hexagon_zigzag" : [6, side_length, 0, (jnp.sqrt(3)+2/jnp.sqrt(3))*a, jnp.sqrt(3)*a, lambda k: jnp.array([0 ,(1-3*(-1)**k)*a/(4*jnp.sqrt(3)),0])]                
                }

    n, side_length, orientation_angle, min_side_length, side_length_increment, shift_vec_for_edge_alignment = mapping[f"{polygon_id}_{edge_type}"]
    polygon=get_polygon(n, side_length, orientation_angle, min_side_length, side_length_increment, shift_vec_for_edge_alignment)     
    path = Path(polygon[:,:2])
    width=jnp.max(polygon[:,0])-jnp.min(polygon[:,0])
    height=jnp.max(polygon[:,1])-jnp.min(polygon[:,1])
    m=jnp.int32(width/(3*a)+2)
    n=jnp.int32(height/(jnp.sqrt(3)*a)+2)
    sheet=get_graphene_sheet(m, n, cc_dist=a)
    inside = path.contains_points(sheet[:,:2],radius=-0.01*a) # Watch the -ve sign here
    polygon = jnp.vstack([polygon[:, :2], polygon[0, :2]])
    flake=remove_single_neighbor_points(sheet[inside],radius=1.01*a)
    sublattice=assign_sublattice(flake,cc_dist=a)
    return m, n,  polygon, flake,  sheet, sublattice
