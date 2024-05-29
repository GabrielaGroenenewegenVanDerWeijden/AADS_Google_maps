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
        
        # Initializing the destination, source and road_grid as grid.
        self.destination = destination
        self.grid = road_grid

        # Calling the main_loop.
        self.main_loop()

        # Returning the path and the length of the path.
        return self.find_path()

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

        # Initializing the path list and the initial current node as the destination.
        path = []
        current = self.destination

        # For as long the node is not None it will be added to the path and the new current node is initialized using the dictionary of history.
        while current is not None:
            path.append(current)
            current = self.history[current]

        # The source is the first node in the history.
        source = list(self.history.keys())[0]

        # Calculating length according to Manhattan distance.
        length = np.absolute (source[0] - self.destination[0]) + np.absolute (source[1] - self.destination[1])

        # Reverse the path to get it from source to destination.
        path = path[::-1]
        
        # Returning the path and the length (as a float).
        return path, float(length)
              
    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        # While the queue is not empty we search the node. 
        while self.queue:
            # Initializing the current node.
            current = self.queue.popleft()

            # Checking if it satisfies the base case.
            if self.base_case(current):
                return

            # Otherwise, we are gonna go through all the next possible steps.
            for new_node in self.next_step(current):
                self.step(current, new_node)
        

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node/coordinate
        :type node: tuple[int]
        :return: This returns if the base case is found or not
        :rtype: bool
        """
    
        # When our node is equal to the destination node. We have completed our path.
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
        # First we need to check if we have already gone through the next node. If not we add it to our queue and add to our history as key and with the value the previous node.
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

        # Looping throught every possible direction and checking if they are possible.
        for direction in [(row + 1, col), (row, col + 1), (row, col - 1), (row - 1, col)]:
            if direction[0] < self.grid.shape[0] and direction[1] < self.grid.shape[1] and direction[0] >= 0 and direction[1] >=0 and self.grid[direction] != 0:
                pos_steps.append(direction)
  
        # Returning the possible steps.
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
        self.find_edges()  # This will be implemented in the next notebook cell.
        
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
            # Initializing the current node.
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

        # If the number of actions is one, then it is a deadend, if the number of actions are 3 or four then it is a junction. Hoewer, if the number
        # of actions are two the node can be an edje or a corner, therefore we check it.
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

        # Getting the indices of the current node.
        row, col = coordinate[0], coordinate[1]

        # Initializing the te list.
        pos_steps = []

        # Looping through all neighbour coordinates and checking if they are within bounds and not a building.
        for neighbour in [(row + 1, col), (row, col + 1), (row, col - 1), (row - 1, col)]:
            if neighbour[0] < self.map.shape[0] and neighbour[1] < self.map.shape[1] and neighbour[0] >= 0 and neighbour[1] >=0 and self.map[neighbour] != 0:
                pos_steps.append(neighbour)
        
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
        # Initializing the stack.
        stack = self.adjacency_list.keys()

        # Looping through each node in the stack.
        for node in stack:

            # Finding all possible neighbours for the node.
            neighbours = self.neighbour_coordinates(node)
    
            # Looping through each direction a neighbour exists.
            for neighbour in neighbours: 
                
                # Getting the speed of the road, by checking the speed of the neighbour, since the speed doesn't change along the road.
                speed = self.map[neighbour]

                # Initialiing the neigbour and the distance, in the direction of the neighbour. 
                neighbour_node, distance = self.find_next_node_in_adjacency_list(node, (neighbour[0] - node[0] , neighbour[1] - node[1]))
                
                # Updating the adjacency list with the neighbour node and distance.
                self.adjacency_list[node].add((neighbour_node, distance, speed))
            
        

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

        # Initializing the current node.
        current_node = node

        # Setting the distance to zero.
        distance = 0

        # While the current node is not a node or not the initial node. We keep going towards the direction.
        while current_node not in self.adjacency_list.keys() or current_node == node:
            distance += 1
            current_node = (current_node[0] + direction[0], current_node[1] + direction[1])
        return current_node, distance

############ CODE BLOCK 120 ################

class FloodFillSolverGraph(FloodFillSolver):
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the nodes that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __call__(self, graph, source, destination):      
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        Hint: The history is already given as a dictionary with as keys the node in the state-space graph and
        as values the previous node from which this node was visited.

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts.
        :type source: tuple[int]
        :param destination: The node where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of nodes and the length of the route.
        :rtype: list[tuple[int]], float
        """       
        self.queue = deque([source])
        self.history = {source: None}

        # Initializing the destination and the graph.
        self.destination = destination
        self.graph = graph

        # Calling the main_loop.
        self.main_loop()

        # Returning the path and the length of the path.
        return self.find_path()       
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """

        # Initializing the te list.
        pos_steps = []

        # Looping throught every possible direction. However, since the graph has also the distance and speed, we take the only the node into consideration.
        for direction in self.graph[node]:
            pos_steps.append(direction[0])
            
        # Returning the possible steps.
        return pos_steps

############ CODE BLOCK 130 ################

class BFSSolverShortestPath():
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the distances it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple(int), float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the distance it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], int]]
    """   
    def __call__(self, graph, source, destination):      
        """
        This method gives the shortest route through the graph from the source to the destination node.
        You start at the source node and the algorithm ends if you reach the destination node, 
        both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts
        :type source: tuple[int] 
        :param destination: The node where the path ends
        :type destination: tuple[int]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: The shortest route and the time it takes. The route consists of a list of nodes.
        :rtype: list[tuple[int]], float
        """       
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.destination = destination

        # Initializing the graph.
        self.graph = graph

        # Calling the main_loop.
        self.main_loop()
    
        # Returning the route and the cost, by using the find_path() function to compute them.
        return self.find_path()

    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        # Initializing the route list and the current node.
        route = []
        current = self.destination

        # For as long the current node is not None it will be added to the route and added to history.
        while current is not None:
            route.append(current)
            current = self.history[current][0]

        # Initializing the time it costs (distance) from that node.
        cost = self.history[self.destination][1]

        # Reverse the path to get it from source to destination.
        route = route[::-1]
        
        # Returning the path and the cost (as a float).
        return route, float(cost)     

    def main_loop(self):
        """
        This method contains the logic of the flood-fill algorithm for the shortest path problem.

        It does not have any inputs nor outputs. 
        Hint, use object attributes to store results.
        """
        # While the priority queue is not empty.
        while self.priorityqueue:
            # We have to sort the priority queue according to the node with the smallest cost (distance).
            self.priorityqueue.sort(key=lambda x: x[1])
            current_node, distance = self.priorityqueue.pop(0)

            # If the base case is satisfied we can return the route and cost.
            if self.base_case(current_node):
                return

            # Otherwise, we are gonna go through all the next possible steps. Plus taking into account the speed_limit and the distance.
            for next_node, distance, speed_limit in self.next_step(current_node):
                # Computing the new cost and calling step.
                new_cost = self.new_cost(current_node, distance, speed_limit)
                self.step(current_node, next_node, new_cost, speed_limit)


    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        #When our node is equal to the destination node. We have completed our path.
        if node == self.destination:
            return True
        return False

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        For now, speed_limit can be ignored.

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        # Initializing the cost of the previous node
        previous_cost = self.history[previous_node][1]
    
        # Returning the new cost.
        return previous_cost + distance

    def step(self, node, new_node, distance, speed_limit):
        """
        One step in the BFS algorithm. For now, speed_limit can be ignored.

        :param node: The current node
        :type node: tuple[int]
        :param new_node: The next node that can be visited from the current node
        :type new_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        """

        # If the new_node is not in history, it is added to the history and the priority queue.
        if new_node not in self.history:
            self.priorityqueue.append((new_node, distance))
            self.history[new_node] = (node, distance)
    
    def next_step(self, node):
        """
        This method returns the next possible actions.

        :param node: The current node
        :type node: tuple[int]
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """

        # Returning the next nodes.
        return self.graph[node]

############ CODE BLOCK 200 ################

class BFSSolverFastestPath(BFSSolverShortestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited 
                              including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the time it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):      
        """
        This method gives a fastest route through the grid from source to destination.

        This is the same as the `__call__` method from `BFSSolverShortestPath` except that 
        we need to store the vehicle speed. 
        
        Here, you can see how we can overwrite the `__call__` method but 
        still use the `__call__` method of BFSSolverShortestPath using `super`.
        """

        self.vehicle_speed = vehicle_speed
        
        return super(BFSSolverFastestPath, self).__call__(graph, source, destination)

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        Use the `speed_limit` and `vehicle_speed` to determine the time/cost it takes to go to
        the new node from the previous_node and add the time it took to reach the previous_node to it..

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        # Initializing the cost of the previous node
        previous_cost = self.history[previous_node][1]

        # The time is the distance divided by the speed_limit of the vehicle speed, depending on which is smaller.
        time = distance / min(speed_limit, self.vehicle_speed)

        # Returning the new cost.
        return previous_cost + time

############ CODE BLOCK 210 ################

def coordinate_to_node(map_, graph, coordinate):
    """
    This function finds a path from a coordinate to its closest nodes.
    A closest node is defined as the first node you encounter if you go a certain direction.
    This means that unless the coordinate is a node, you will need to find two closest nodes.
    If the coordinate is a node then return a list with only the coordinate itself.

    :param map_: The map of the graph
    :type map_: Map
    :param graph: A Graph of the map
    :type graph: Graph
    :param coordinate: The coordinate from which we want to find the closest node in the graph
    :type coordinate: tuple[int]
    :return: This returns a list of closest nodes which contains either 1 or 2 nodes.
    :rtype: list[tuple[int]]
    """

    # If the coordinate is a node then we return the coordinate in a list. Otherwise, we'll check for the closest nodes.
    if coordinate in graph:
        return [coordinate]
    
    # Initializing the grid.
    grid = map_.grid

    # Initializing the list of the closest nodes.
    closest = []

    # Initializing the possible directions.
    direction = [(1,0), (-1, 0), (0, 1), (0, -1)]

    # Going through each direction.
    for way in direction:
        # We initialize a boolean until we have found the node and aftwer each direction we must reinitialize the coordinate.
        Found = False
        new_coordinate = coordinate
        while not Found:
            
            new_coordinate = (new_coordinate[0] + way[0], new_coordinate[1] + way[1])
            
            # Checking if the new_coordinare is valid.
            if new_coordinate[0] < grid.shape[0] and new_coordinate[1] < grid.shape[1] and new_coordinate[0] >= 0 and new_coordinate[1] >=0 and grid[new_coordinate] > 0:
                # If the new_coordinate is in the graph it means that it is a  node and therefor we can start the loop againg for another way.
                if new_coordinate in graph:
                    closest.append(new_coordinate)
                    Found = True
            else:
                break

    # Returning the closest nodes.
    return closest

############ CODE BLOCK 220 ################

def create_country_graphs(map_):
    """
    This function returns a list of all graphs of a country map, where the first graph is the highways and de rest are the cities.

    :param map_: The country map
    :type map_: Map
    :return: A list of graphs
    :rtype: list[Graph]
    """
    # Initializing the city corners.
    city_corners = map_.city_corners
    
    # Getting the highway map.
    highway_map = map_.get_highway_map()

    # Getting the city map withoud the highways.
    map_city = map_.get_city_map()

    # Initializing the city graphs list.
    city_graphs = []
    for corner in city_corners:
        city_graphs.append(Graph(map_city, start=corner))
        
    return [Graph(highway_map)] + city_graphs

############ CODE BLOCK 230 ################

class BFSSolverMultipleFastestPaths(BFSSolverFastestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that are visited and as values the node that leads to this node including the time it takes from the start node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
        :param found_destinations: The destinations already found with Dijkstra.
        :type found_destinations: list[tuple[int]]
    """
    def __init__(self, find_at_most=3):
        """
        This init makes it possible to make a different Dijkstra algorithm 
        that find more or less destination nodes before it stops searching.

        :param find_at_most: The number of found destination nodes before the algorithm stops
        :type find_at_most: int
        """
        self.find_at_most = find_at_most
    
    def __call__(self, graph, sources, destinations, vehicle_speed):      
        """
        This method gives the top three fastest routes through the grid from any of the sources to any of the destinations.
        You start at the sources and the algorithm ends if you reach enough destinations, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param sources: The nodes where the path starts and the time it took to get here.
        :type sources: list[tuple[tuple[int], float]]
        :param destinations: The nodes where the path ends and the time it took to get here.
        :type destinations: list[tuple[tuple[int], float]]
        :param vehicle_speed: The maximum speed of the vehicle.
        :type vehicle_speed: float
        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """       
        self.priorityqueue = sorted(sources, key=lambda x:x[1])
        self.history = {s: (None, t) for s, t in sources}
        
        self.destinations = destinations
        self.destination_nodes = [dest[0] for dest in destinations]
        self.found_destinations = []

        raise NotImplementedError("Please complete this method")       

    def find_n_paths(self):
        """
        This method needs to find the top `n` fastest paths between any source node and any destination node.
        This does not mean that each source node has to be in a path nor that each destination node needs to be in a path.

        Hint1: The fastest path is stored in each node by linking to the previous node. 
               Therefore, if you start searching from a destination node,
               you always find the optimal path from that destination node.
               This is similar if you only had one destination node.         

        :return: A list of the n fastest paths and time they take, sorted from fastest to slowest 
        :rtype: list[tuple[path, float]], where path is a fictional data type consisting of a list[tuple[int]]
        """
        raise NotImplementedError("Please complete this method")       
        
    def base_case(self, node):
        """
        This method checks if the base case is reached and
        updates self.found_destinations

        :param node: The current node
        :type node: tuple[int]
        :return: Returns True if the base case is reached.
        :rtype: bool
        """
        raise NotImplementedError("Please complete this method")

############ CODE BLOCK 235 ################

class BFSSolverFastestPathMD(BFSSolverFastestPath):
    def __call__(self, graph, source, destinations, vehicle_speed):      
        """
        This method is functionally no different than the call method of BFSSolverFastestPath
        except for what `destination` is.

        See for an explanation of all arguments `BFSSolverFastestPath`.
        
        :param destinations: The nodes where the path ends.
        :type destinations: list[tuple[int]]
        """
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.destinations = destinations
        self.destination = None
        self.vehicle_speed = vehicle_speed   
    
        # Initializing the graph.
        self.graph = graph

        # Calling the main_loop.
        self.main_loop()

        # Retuning the path and cost.
        return self.find_path()
        

    def base_case(self, node):
        """
        This method checks if the base case is reached.

        :param node: The current node
        :type node: tuple[int]
        :return: returns True if the base case is reached.
        :rtype: bool
        """
        # If the node is one of the destinations we return True. Otherwise, False.
        if node in self.destinations:
            self.destination = node
            return True
        return False

############ CODE BLOCK 300 ################

def path_length(coordinate, closest_nodes, map_, vehicle_speed):
    return [(node, (abs(node[0] - coordinate[0]) + abs(node[1] - coordinate[1])) / min(vehicle_speed, map_[coordinate])) for node in closest_nodes] 

def find_path(coordinate_A, coordinate_B, map_, vehicle_speed, find_at_most=3):
    """
    Find the optimal path according to the divide and conquer strategy from coordinate A to coordinate B.

    See hints and rules above on how to do this.

    :param coordinate_A: The start coordinate
    :type coordinate_A: tuple[int]
    :param coordinate_B: The end coordinate
    :type coordinate_B: tuple[int]
    :param map_: The map on which the path needs to be found
    :type map_: Map
    :param vehicle_speed: The maximum vehicle speed
    :type vehicle_speed: float
    :param find_at_most: The number of routes to find for each path finding algorithm, defaults to 3. 
                         Note, that this is only needed if you did 2.3.
    :type find_at_most: int, optional
    :return: The path between coordinate_A and coordinate_B. Also, return the cost.
    :rtype: list[tuple[int]], float
    """

    # Finding the closest nodes A and B to the coordinates of A and B.
    closest_nodes_A = coordinate_to_node(map_ = map_, graph = Graph(map_), coordinate = coordinate_A)
    closest_nodes_B = coordinate_to_node(map_ = map_, graph = Graph(map_), coordinate = coordinate_B)

    # Initializgin the distance from the coordinate to the nodes.
    distances_A = path_length(coordinate_A, closest_nodes_A, map_, vehicle_speed)
    distances_B = path_length(coordinate_B, closest_nodes_B, map_, vehicle_speed)

    # Finding the closest.
    closest_A = min(distances_A, key = lambda x: x[1])
    closest_B = min(distances_B, key = lambda x: x[1])

    # Initializing the highway graph and a list of the city graphs.
    highway_graph, *city_graphs = create_country_graphs(map_)

    # Searching in which city each coordinate is.
    for i, city in enumerate(city_graphs):
        if closest_A[0] in city:
            city_A = i
        if closest_B[0] in city:
            city_B = i
    
    # Getting all the city exits.
    exits = map_.get_all_city_exits()

    # Finding the closest exits to the starting nodes
    path_A_exit, cost_A_exit =  BFSSolverFastestPathMD()(city_graphs[city_A], closest_A[0], exits, vehicle_speed)
    exit_A = path_A_exit[-1]
    path_B_exit, cost_B_exit=  BFSSolverFastestPathMD()(city_graphs[city_B], closest_B[0], exits, vehicle_speed)
    exit_B = path_B_exit[-1]

    # Calculating the distance between the exit nodes.
    exits_path, exits_cost = BFSSolverFastestPath()(highway_graph, exit_A, exit_B, vehicle_speed)

    # Now adding the three parts together.
    total_path = path_A_exit + exits_path + path_B_exit[::-1]
    # Removing all duplicates.
    total_path = list(dict.fromkeys(total_path))

    # Now we need to add the distance between the coordinate to their closest nodes.
    total_cost = cost_A_exit + exits_cost + cost_B_exit + closest_A[1] + closest_B[1] 

    # Looking if the coordinates are in the same city. If they are we calculate the fastest path.
    if city_A == city_B:
        if len(closest_nodes_A) == 1:
            path_city, cost_city =  BFSSolverFastestPathMD()(city_graphs[city_A], closest_A[0], closest_nodes_B, vehicle_speed)

        elif len(closest_nodes_B) == 1:
            path_city, cost_city =  BFSSolverFastestPathMD()(city_graphs[city_A], closest_B[0], closest_nodes_A, vehicle_speed)

        else:
            path1, cost1 =  BFSSolverFastestPathMD()(city_graphs[city_A], closest_nodes_A[0], closest_nodes_B, vehicle_speed)
            path2, cost2 =  BFSSolverFastestPathMD()(city_graphs[city_A], closest_nodes_A[1], closest_nodes_B, vehicle_speed)

            if cost1 < cost2:
                path_city = path1
                cost_city = cost1
            path_city = path2
            cost_city = cost2
        
        # Cheking if going through the city is faster than going through the highway.
        if cost_city < total_cost:
            return path_city, cost_city

    return total_path, total_cost


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
