############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

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


        # Looping through the queue until it is empty.
        while queue:
            # In the first loop the current node would be 'self.start', from there after each iteration it would be the next node in the queue.   
            current = queue.popleft()

            # Finding the possible actions using the neighbour_coordinates function.
            actions = self.neighbour_coordinates(current)

            # Checking if the current node needs to be added to the adjacency list.
            self.adjacency_list_add_node(current, actions)

            # Going throught all the actions and adding them to the queue and history if they are not in history. They should be added in the queue and history.
            for action in actions:
                if action not in history:
                    queue.append(action)
                    history.add(action)
            

                    
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
        if n_actions != 2 or (actions[0][0] != actions[1][0] and actions[0][1] != actions[1][1]):
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

        # Looping throught every possible direction and checking if they are possible.
        for direction in [(row + 1, col), (row, col + 1), (row, col - 1), (row - 1, col)]:
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

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """

        print(map_.grid)
        # Getting the adjacency list.
        print("Possible previous", graph)
        adj_list = graph.adjacency_list

        # Initializing the list of the four possible directions
        direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
       
        for node in adj_list.keys():
            # Finding each neighbour node of the current node by checking every direction. 
            for way in direction:
                neighbour_node, distance, speed = self.find_next_node_in_adjacency_list(node, way)
                # If the neighbour isn't none, (meaning that diststance and speed aren't none either)then we can add them in the adjacency list and vice versa.
                if neighbour_node is not None: 
                    adj_list[node].add((neighbour_node, distance, speed))
                    #adj_list[neighbour_node].add((node, distance))

        for i,k in adj_list.items():
            print(i,k)


    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """

        # Initializing the grid.
        grid = map_.grid

        # Initializing the node.
        current_node = node

        # Getting the adjacency list
        adj_list = graph.adjacency_list

        # Setting the distance to 0
        distance = 0

        # Initializing the list to keep track of the speed limits.
        speed = []

        # Initializing GO
        True

        # We keep changing the coordinates of the code towards the direction as long as it stays within the grid.
        while True:
            # Calculating the new coordinates of the current node when following the direction
            current_node = (current_node[0] + direction[0], current_node[1] + direction[1])
            speed.append(grid[current_node])

            if not (current_node[0] < grid.shape[0] - 1 and current_node[1] < grid.shape[1] - 1 and current_node[0] >= 0 and current_node[1] >= 0 and grid[current_node] != 0):
                break
            
            # Adding the distance.
            distance += 1

            # Checking we have reached the next by checking if the coordinates are in the adjacency list.
            if current_node in adj_list.keys():
                neighbour_node = current_node

                # Calculating the mode
                speed_limit = max(speed, key = speed.count)

                return neighbour_node, distance, speed_limit
           
        # Otherwise there is no neighbour node. And return none?
        return None, None, None


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
