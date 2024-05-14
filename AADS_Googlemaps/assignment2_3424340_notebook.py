############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 1 ################

class FloodFillSolver():
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the coordinates that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    
    def __call__(self, road_grid, source, destination):
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both coordinates should be included in the path.
        To find the shortest route a version of a flood fill algorithm is used, see the explanation above.
        A route consists of a list of coordinates.

        Hint: The history is already given as a dictionary with as keys the coordinates in the state-space graph and
        as values the previous coordinate from which this coordinate was visited.

        :param road_grid: The array containing information where a house (zero) or a road (one) is.
        :type road_grid: np.ndarray[(Any, Any), int]
        :param source: The coordinate where the path starts.
        :type source: tuple[int]
        :param destination: The coordinate where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of coordinates and the length of the route.
        :rtype: list[tuple[int]], float
        """
        self.queue = deque([source])
        self.history = {source: None}
        
        # Initializing the destination, source and road_grid.
        self.destination = destination
        self.source = source
        self.road_grid = road_grid

        # Calling the main_loop.
        self.main_loop()

        # Calculating the path and the length of the path.
        path, length = self.find_path()

        # Returning the path and the length of the path.
        return path, length

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one coordinate to the next has a length of 1.
        For example: The distance between coordinates (0,0) and (0,1) is 1 and 
                     The distance between coordinates (3,0) and (3,3) is 3. 

        The distance is the Manhattan distance of the path.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        path = []
        current = self.destination
        while current is not None:
            path.append(current)
            current = self.history[current]

         # Calculating length according to Manhattan distance.
        length = np.absolute (self.source[0] - self.destination[0]) + np.absolute (self.source[1] - self.destination[1])

        # Reverse the path to get it from source to destination.
        path = path[::-1]
        
        # Returning the path and the length.
        return path, length
              
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        while self.queue:
            node = self.queue.popleft()
            if self.base_case(node):
                return

            for new_node in self.next_step(node):
                self.step(node, new_node)

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the base case is found or not
        :rtype: bool
        """
        # The base case is that the current node is the destination. Thus when 
        # we return true we know we have found the destination.
        if node == self.destination:
            return True

        return False
        
        
    def step(self, node, new_node):
        """
        One flood-fill step.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :param new_node: The next node/coordinate that can be visited from the current node/coordinate
        :type new_node: tuple[int]       
        """
        # First we need to check if we have already gond through the next node. If not we add it to our queue and add to our history as key and with the value the previous node.
        if new_node not in self.history:
            self.queue.append(new_node)
            self.history[new_node] = node

    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """

        # Getting the indices of the current node.
        row, col = node[0], node[1]

        # Initializing the te list.
        pos_steps = []

        for direction in [(row + 1, col), (row, col + 1), (row, col - 1), (row - 1, col)]:
            # check if valid
            if direction[0] < self.road_grid.shape[0] and direction[1] < self.road_grid.shape[1] and direction[0] >= 0 and direction[1] >=0 and self.road_grid[direction] != 0:
                pos_steps.append(direction)
            
        return pos_steps

############ CODE BLOCK 10 ################

class GraphBluePrint():
    """
    You can ignore this class, it is just needed due to technicalities.
    """
    def find_nodes(self): pass
    def find_edges(self): pass
    
class Graph(GraphBluePrint):   
    """
    Attributes:
        :param adjacency_list: The adjacency list with the road distances and speed limit.
        :type adjacency_list: dict[tuple[int]: set[edge]], where an edge is a fictional datatype 
                              which is a tuple containing the datatypes tuple[int], int, float
        :param map: The map of the graph.
        :type map: Map
    """
    def __init__(self, map_, start=(0, 0)):
        """
        This function transforms any (city or lower) map into a graph representation.

        :param map_: The map that needs to be transformed.
        :type map_: Map
        :param start: The start node from which we will find all other nodes.
        :type start: tuple[int]
        """
        self.adjacency_list = {}
        self.map = map_
        self.start = start
        
        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell
        
    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        So far, we called this method `step`. However, this class is more than just the search algorithm,
        therefore, we gave it a bit more descriptive name.

        Note, that we only want to find the nodes, so history does not need to contain a partial path (previous node).
        In `find_edges` (the next cell), we will add edges for each node.
        """
        queue = deque([self.start])
        history = {self.start}

        # Initializing the starting node to be the current node.
        current = self.start

        if current not in history:
            self.queue.append(current)
            self.history.add(current)
        
        
        # While the queue is not empty
        while queue:
            if cur_node not in history:
                cur_node = self.queue.popleft()

            actions = self.neighbour_coordinates(node)
            self.adjacency_list_add_node(node, actions)

        #raise NotImplementedError("Please complete this method")
                    
    def adjacency_list_add_node(self, coordinate, actions):
        """
        This is a helper function for the breadth-first search algorithm to add a coordinate to the `adjacency_list` and
        to determine if a coordinate needs to be added to the `adjacency_list`.

        Reminder: A coordinate should only be added to the adjacency list if it is a corner, a crossing, or a dead end.
                  Adding the coordinate to the adjacency_list is equivalent to saying that it is a node in the graph.

        :param coordinate: The coordinate that might need to be added to the adjacency_list.
        :type coordinate: tuple[int]
        :param actions: The actions possible from this coordinate, an action is defined as an action in the coordinate state-space.
        :type actions: list[tuple[int]]
        """

        # Determining the number of actions.
        n_actions = len(actions)

        # Accourding to the ammount of possible actions we can deduce whether the coordinate is a cross node. 1 meaning it a deadend, 3 & 4 cross node/junction node. 
        # Hoeweveer, if it's 2 doesn't neccesarily mean it a corner because it can be an edge as well. Thus, we specify if its indeed a corner.
        if n_actions != 2 or not (action[0][0] == actions[1][0] and action[0][1] == actions[1][1]):
            self.adjacency_list[coordinate] = set()
        
                           
    def neighbour_coordinates(self, coordinate):
        """
        This method returns the next possible actions and is part of the breadth-first search algorithm.
        Similar to `find_nodes`, we often call this method `next_step`.
        
        :param coordinate: The current coordinate
        :type coordinate: tuple[int]
        :return: A list with possible next coordinates that can be visited from the current coordinate.
        :rtype: list[tuple[int]]  
        """
    
        # Mostly the same function as the flood fill above.

        # Getting the indices of the current node.
        row, col = coordinate[0], coordinate[1]

        # Initializing the te list.
        pos_steps = []

        # Getting the grid.
        grid = map_.grid

        for direction in [(row + 1, col), (row, col + 1), (row, col - 1), (row - 1, col)]:
            # check if valid
            if direction[0] < grid.shape[0] and direction[1] < grid.shape[1] and direction[0] >= 0 and direction[1] >=0 and grid[direction] != 0:
                pos_steps.append(direction)
            
        return pos_steps   
    
    def __repr__(self):
        """
        This returns a representation of a graph.

        :return: A string representing the graph object.
        :rtype: str
        """
        # You can change this to anything you like, such that you can easily print a Graph object. An example is already given.
        return repr(dict(sorted(self.adjacency_list.items()))).replace("},", "},\n")

    def __getitem__(self, key):
        """
        A magic method that makes using keys possible.
        This makes it possible to use self[node] instead of self.adjacency_list[node]

        :return: The nodes that can be reached from the node `key`.
        :rtype: set[tuple[int]]
        """
        return self.adjacency_list[key]

    def __contains__(self, key):
        """
        This magic method makes it possible to check if a coordinate is in the graph.

        :return: This returns if the coordinate is in the graph.
        :rtype: bool
        """
        return key in self.adjacency_list

    def get_random_node(self):
        """
        This returns a random node from the graph.
        
        :return: A random node
        :rtype: tuple[int]
        """
        return tuple(RNG.choice(list(self.adjacency_list)))
        
    def show_coordinates(self, size=5, color='k'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.

        :param size: The size of the dots, default to 5
        :type size: int
        :param color: The Matplotlib color of the dots, defaults to black
        :type color: string
        """
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)        

    def show_edges(self, width=0.05, color='r'):
        """
        If this method is used before another method that does a plot, it will be plotted on top.
        
        :param width: The width of the arrows, default to 0.05
        :type width: float
        :param color: The Matplotlib color of the arrows, defaults to red
        :type color: string
        """
        for node, edge_list in self.adjacency_list.items():
            for next_node,_,_ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1])*0.975, (next_node[0] - node[0])*0.975, color=color, length_includes_head=True, width=width, head_width=4*width)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
