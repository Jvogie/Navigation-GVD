from pathlib import Path
from queue import PriorityQueue
from typing import Set, Tuple, List

import numpy as np
import numpy.typing as npt

from hw1.utils import neighbors, plot_GVD, PathPlanMode, distance


def cell_to_GVD_gradient_ascent(
    grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the shortest path from any cell in the enviroment to a cell on the
    GVD using gradient ascent.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set[tuple]): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list<tuple>: list of tuples of the path.
    """

    path = [cell]

# initially set the highest value to the starting/ending cell of path
    high_value = cell

# loop will run until we hit a value in GVD
    while high_value not in GVD:
        # setting the first and second values of the tuple to an
        # individual variable so that i can use them in neighbors function
        cell_i, cell_j = high_value
        # iterate through each neighbor that neighbor function gives me and check if they are
        # than current highest value. If so set to high_value variable. Append each iteration of
        # high_value to path list and return path once we hot GVD and while loop is broken
        for neighbor in neighbors(grid, cell_i, cell_j):
            if neighbor > high_value:
                high_value=neighbor

            path.append(high_value)

    return path


def cell_to_GVD_a_star(
    grid: npt.ArrayLike, GVD: Set[Tuple[int, int]], cell: Tuple[int, int], 
    goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Find the shortest path from any cell in the enviroment to the GVD using
    A* with L2 distance heurstic.
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        GVD (set<tuple>): A set of tuples containing the cells on the GVD.
        cell (tuple): The starting/ending cell of this path.
    Returns:
        list[tuple], dict, list[int]: list of tuples of the path, and the reached 
        dictionary, and the list of frontier sizes. 
    """

    # define a priority queue
    frontier = PriorityQueue()
    frontier.put((0, cell))
    frontier_size = [0]

    # construct a reached table using python dictionary. The key is the (x, y)
    # tuple of the cell position, the value is dictiionary with the cell's cost,
    # and cell parent.
    reached = {cell: {"cost": 0, "parent": None}}

    # created a while loop that lasts until frontier is empty or a path to GVD is found

    while not frontier.empty():
        # intialized current value from frontier and i and j coordinates from current tuple
        low_f, current = frontier.get()
        i_cord, j_cord = current
        # check if we have hit GVD if current tuple is in GVD set. If so create a path to cell
        #  by going back through parent history in reached table tell we hit given start/end point.
        if current in GVD:
            path=[]
            while current !=cell:
                path.append(current)
                current = reached[current]["parent"]
            path.append(cell)
            return path[::-1], reached, frontier_size

        # for loop to go through each neighbor around current cell. Created cost function that increases
        # based on distance from current cell to neighbor.
        for neighbor in neighbors(grid, i_cord, j_cord):
            cost = reached[current]["cost"]+distance(current, neighbor)
            # check to see if neighbor is in reached table or if current cost is less than neighbor cost
            # if check is passed, we add neighbor to reached table with cost and parent.
            # Then calculated f function where g = cost and h = neighbor distance to goal and set
            # it to the priority variable. This was then put into the frontier
            if neighbor not in reached or cost < reached[neighbor]["cost"]:
                reached[neighbor] = {"cost": cost, "parent":current}
                priority = cost + distance(neighbor, goal)
                frontier.put((priority, neighbor))
                frontier_size.append(frontier.qsize())

    return None, reached, frontier_size


def GVD_path(
    grid: npt.ArrayLike,
    GVD: Set[Tuple[int, int]],
    A: Tuple[int, int],
    B: Tuple[int, int],
    mode: PathPlanMode
) -> List[Tuple[int, int]]:
    """Find the shortest path between two points on the GVD using
    Breadth-First-Search and DFS
    Args:
        grid (numpy): NxN numpy array representing the world, with obstacles,
        walls, and the distance from each cell to the obstacles.
        A (tuple): The starting cell of the path.
        B (tuple): The ending cell of the path.
    Returns:
        list[tuple], dict, list[int]: return the path, pointers, and frontier 
        size array. 
    """

    # the set of cells on the GVD
    GVD = set(GVD)

    # the set of visited cells
    closed = set([])

    # the set of cells on the current frontier
    frontier = [A]

    # back pointers to find the path once reached the goal B. The keys
    # should both be tuples of cell positions (x, y)
    pointers = {}

    # the length of the frontier array, update this variable at each step. 
    frontier_size = [0]

    # while loop that runs as long as frontier is not empty. Check if mode is DFS or BFS.
    # Intiate while loop that goes through frontier and pops a node which is then set to current variable
    # with the i and j coordinates pulled out from it to use in neighbors function. While loop breaks if current
    # node is ever equal to B/goal tuple.
    while len(frontier) > 0:
        if mode == PathPlanMode.DFS:
            pointers[A] = None
            while frontier:
                current=frontier.pop()
                i_cord, j_cord = current
                if current == B:
                    break
                # For each neighbor of current node check if they are not in pointers and in GVD.
                # If so add each neighbor to frontier and add current frontier size to frontier_size list
                # Add each neighbor to closed list, so we know what cells we visited. Break for loop once we find B
                for neighbor in neighbors(grid, i_cord, j_cord):
                    if neighbor not in pointers and neighbor in GVD:
                        frontier.append(neighbor)
                        pointers[neighbor] = current
                        frontier_size.append(len(frontier))
                        closed.add(neighbor)
                        if neighbor == B:
                            break
            # Once we find B, initialize path that starts at B and goes back through each pointer and
            # adds it to path list. Once we reach point A, reverse the list and we have our path from A to B
            path = []
            if B in pointers:
                node=B
                while node !=A:
                    path.append(node)
                    node= pointers[node]
                path.append(A)
                path.reverse()
            else:
                path = None

        # BFS was implemented the same as above but we always pop the first element in our frontier
        elif mode == PathPlanMode.BFS:
            pointers[A] = None
            while frontier:
                current = frontier.pop(0)
                i_cord, j_cord = current
                if current == B:
                    break
                for neighbor in neighbors(grid, i_cord, j_cord):
                    if neighbor not in pointers and neighbor in GVD:
                        frontier.append(neighbor)
                        pointers[neighbor] = current
                        frontier_size.append(len(frontier))
                        closed.add(neighbor)
                        if neighbor == B:
                            break
            path = []
            if B in pointers:
                node= B
                while node != A:
                    path.append(node)
                    node = pointers[node]
                path.append(A)
            else:
                path = None

    return path, pointers, frontier_size

def compute_path(
    grid,
    GVD: set[tuple],
    start: tuple,
    goal: tuple,
    outmode: PathPlanMode = PathPlanMode.GRAD,
    inmode: PathPlanMode = PathPlanMode.DFS):

    """ Compute the path on the grid from start to goal using the methods
    implemented in this file. 
    Returns:
        list: a list of tuples represent the planned path. 
    """

    if outmode == PathPlanMode.GRAD:
        start_path = cell_to_GVD_gradient_ascent(grid, GVD, start)
        end_path = list(reversed(cell_to_GVD_gradient_ascent(grid, GVD, goal)))
    else:
        start_path = cell_to_GVD_a_star(grid, GVD, start, goal)[0]
        end_path = list(reversed(cell_to_GVD_a_star(grid, GVD, goal, start)[0]))
    mid_path, reached, frontier_size = GVD_path(
        grid, GVD, start_path[-1], end_path[0], inmode)
    return start_path + mid_path[1:-1] + end_path


def test_world(
    world_id, 
    start, 
    goal,
    outmode: PathPlanMode = PathPlanMode.GRAD,
    inmode: PathPlanMode = PathPlanMode.DFS,
    world_dir="worlds"):

    print(f"Testing world {world_id} with modes {inmode} and {outmode}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")
    GVD = set([tuple(cell) for cell in np.load(
        f"{world_dir}/world_{world_id}_gvd.npy")])
    path = compute_path(grid, GVD, start, goal, outmode=outmode, inmode=inmode)
    print(f"Path length: {len(path)} steps")
    plot_GVD(grid, world_id, GVD, path)
