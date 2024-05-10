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
        
        self.destination = destination
        self.road_grid = road_grid
        self.main_loop()
        path = self.find_path()
        length = len(path) - 1  # Length is the number of steps, which is the number of coordinates minus 1
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
        current = destination
        while current is not None:
            path.append(current)
            current = self.history[current]
        return path[::-1]  # Reverse the path to get it from source to destination
              
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
        x_cord, y_cord = node

        # Next possible coordinates
        if y_cord + 1 <= len(self.road_grid.shape(1)):
            right_step = (x_cord, y_cord + 1)
        else:
            right_step = None
        if y_cord - 1 >= 0: 
            left_step = (x_cord, y_cord - 1)
        else:
            left_step = None
        if x_cord + 1 <= len(self.road_grid.shape(0)):
            down_step = (x_cord + 1, y_cord)
        else:
            down_step = None
        if x_cord - 1 >= 0:
            up_step = (x_cord - 1, y_cord)
        else:
            up_step = None
        
        list_of_pos_steps = (right_step, left_step, down_step, up_step)

        for step in list_of_pos_steps:
            if self.road_grid[step] > 0 and step not in self.history.key() and step not in self.queue and step is not None: 
                self.queue.add(step)

        return self.queue
        
    


        

plt.matplotlib.rcParams['figure.dpi'] = max(30, map_.size ** 0.5 // 2)  # Number of pixels, therefore, the quality of the image. A large dpi is very slow.

# Generate a random start and end position in the grid for proper testing.
start = (0,0)
end =  (map_.shape[0]-1, map_.shape[1]-1)

path, length = FloodFillSolver()(map_, start, end)
print(f"The path length was {length}.")
map_.show(path, True)


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
