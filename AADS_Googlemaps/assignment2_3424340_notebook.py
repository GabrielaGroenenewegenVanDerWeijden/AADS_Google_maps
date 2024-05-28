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

        # According to the ammount of possible actions we can deduce whether the coordinate is a cross node. 1 meaning it a deadend, 3 & 4 cross node/junction node. 
        # Hoewever, if it's 2 doesn't neccesarily mean it a corner because it can be an edge as well. Thus, we specify if its indeed a corner.
        # We do this by checking if the two actions aren't on the same row and that the two actions aren't on the same column.
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
            # Finding all the edges in the direction of the neighbours of the node.
            for neighbour in neighbours: 
                # Initializing the speed of the road by checking the speed on the conde of the edge. Since its stated above that we can assume that the speed limit
                # does not change in the entirety of the road.
                speed = self.map[neighbour]
                # Computing the neigbour node and the distance of the node. The direction is calculated by the difference in the coordinates of the neighbour node and the node.
                neighbour_node, distance = self.find_next_node_in_adjacency_list(node, (neighbour[0] - node[0] , neighbour[1] - node[1]))
                # Updating the adjacency list with the neighbour node and distance.
                self.adjacency_list[node].add((neighbour_node, distance, speed))
            
    
        # Om te checken.
        # for i,k in self.adjacency_list.items():
        #      print(i,k)
        

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

        # We initilize
        current_node = node

        # Since node itself is one further we initialize 1
        distance = 0

        # While the current_node is not found in the adjacency means that is not a node and we have to check for the first instance that the current node is definetly node.
        # Thus adding current_node == node, after that current_node gets updated.
        while current_node not in self.adjacency_list.keys() or current_node == node:
            # Calculating the new coordinates of the current node when following the direction
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

        # Calculating the path and the length of the path.
        path, length = self.find_path()

        # Returning the path and the length of the path.
        return path, length     

    #def find_path(self):
    #     """
    #     This method finds the shortest paths between the source node and the destination node.
    #     It also returns the length of the path. 
        
    #     Note, that going from one node to the next has a length of 1.

    #     :return: A path that is the optimal route from source to destination and its length.
    #     :rtype: list[tuple[int]], float
    #     """
    #     raise NotImplementedError("Please complete this method")       

    
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

        # Looping throught every possible direction. Hoever, since the graph has also the distance and speed, we take the only the node into consideration.
        for direction in list(self.graph[node]):
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
        # Initializing the route list and the initial current node as the destination.
        route = []
        current = self.destination

        # For as long the node is not None it will be added to the route and the new current node is initialized using the dictionary of history.
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
        # The base case is that the current node is the destination. Thus when 
        # we return true we know we have found the destination.
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

        # Calculate the new cost (as a float) as the sum of the previous cost and the distance.
        new_cost = float(previous_cost + distance)
    
        # Returning the new cost.
        return new_cost

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

        # First we need to check if we have already gone through the next node. If not we add it to our queue and add to our history as key and with the value the previous node.
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

        # First we need to check if the vehicle's speed is greater than the speed limit. Otherwise, it is useless to drive on a road where we cannot drive to the speed limit.
        time = distance / min(speed_limit, self.vehicle_speed)

        # Calculating the new cost, by summing up the previous cost and the time the vehicle takes to ride that distance. 
        new_cost = float(previous_cost + time)

        # Returning the new cost.
        return new_cost

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
    # Initializing the grid.
    grid = map_.grid

    # Initializing the possible directions.
    direction = [(1,0), (-1, 0), (0, 1), (0, -1)]

    # Initializing the list of the closest nodes.
    closest = []

    # If the coordinate is a node then we return the coordinate in a list.
    if coordinate in graph:
        closest.append(coordinate)
        return closest
    else:
        # We have to look in all directions where the two closest nodes are.
        for way in direction:
            # We initialize a boolean until we have found the node and aftwer each direction we must reinitialize the coordinate.
            Found = False
            new_coordinate = coordinate
            while not Found:
                # Going towards the certain direction.
                new_coordinate = (new_coordinate[0] + way[0], new_coordinate[1] + way[1])
               
                # Checking if valid, otherwise we break the loop and start againg looping towards another direction.
                if new_coordinate[0] < grid.shape[0] and new_coordinate[1] < grid.shape[1] and new_coordinate[0] >= 0 and new_coordinate[1] >=0 and grid[new_coordinate] > 0:
                    # If the new_coordinate is in the graph it means that it is a  node.
                    if new_coordinate in graph:
                        # Appending it to the list.
                        closest.append(new_coordinate)
                        # Changing the boolean Found to True. To start searching in the other direction.
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

    map_city = map_.get_city_map()

    # Initializing the city graphs list.
    city_graphs = []
    for corner in city_corners:
        city_graphs.append(Graph(map_city, start=corner))
        
    return [Graph(highway_map)] + city_graphs

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
        #self.destination = None
        self.vehicle_speed = vehicle_speed   
    
        # Initializing the graph.
        self.graph = graph

        # Calling the main_loop.
        self.main_loop()

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


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
