# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import queue
import math

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


#bfs adding last node twice
def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_states_explored = 0
    start = maze.getStart()
    q = queue.Queue()
    q.put(start)

    visited = {}

    predecessors = {}
    predecessors[start] = None

    while (not q.empty()):

        num_states_explored += 1

        first = q.get()
        current_pred = first
        visited[first] = 1
        neighbors = maze.getNeighbors(first[0], first[1])

        for neighbor in neighbors:
            if (not neighbor in visited) and (maze.isValidMove(neighbor[0], neighbor[1])):
                predecessors[neighbor] = current_pred
                visited[neighbor] = 1
                q.put(neighbor)

    dots = maze.getObjectives()

    goal = maze.getObjectives()[0]
    path = []
    path.append(goal)
    current = goal

    while (current != start):
        path.append(predecessors[current])
        current = predecessors[current]

    path.append(start)
    print(path)
    return path, num_states_explored


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_states_explored = 0
    start = maze.getStart()
    q = queue.LifoQueue()
    q.put(start)

    visited = {}
    predecessors = {}
    predecessors[start] = None

    while (not q.empty()):

        num_states_explored += 1
        first = q.get()
        current_pred = first
        visited[first] = 1
        neighbors = maze.getNeighbors(first[0], first[1])

        for neighbor in neighbors:
            if (not neighbor in visited) and (maze.isValidMove(neighbor[0], neighbor[1])):
                predecessors[neighbor] = current_pred
                visited[neighbor] = 1
                q.put(neighbor)

    dots = maze.getObjectives()

    goal = maze.getObjectives()[0]
    path = []
    path.append(goal)
    current = goal

    while (current != start):
        path.append(predecessors[current])
        current = predecessors[current]

    path.append(start)

    return path, num_states_explored


def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    closedSet = {}
    openSet = {}
    cameFrom = {}

    gScore = {} #distance from start to current node
    fScore = {} #distance from start to current node + man dist of current node to end node

    startNode = maze.getStart()
    #print("startNode")
    #print(startNode)
    endNode = maze.getObjectives()[0]
   # print("endNode")
   # print(endNode)

    fScore[startNode] = manhattan_dist(startNode, endNode)
    num_states_explored = 1

    openSet[startNode] = manhattan_dist(startNode, endNode)

    while (len(openSet.keys()) != 0):
        current = get_min_f_score(openSet, fScore)
        if (current == endNode):
            return reconstruct_path(cameFrom, current), num_states_explored
        
        #print("openset")
        #print(openSet)
        #print("current")
        #print(current)
        closedSet[current] = openSet.pop(current)

        for neigh in maze.getNeighbors(current[0], current[1]):
            if (maze.isValidMove(neigh[0], neigh[1])):
                if (neigh in closedSet): #we already explored this node
                    continue
                else:
                    
                    if (neigh not in openSet):
                        openSet[neigh] = manhattan_dist(neigh, endNode)
                        num_states_explored += 1
                    
                    cameFrom[neigh] = current
                    fScore[neigh] = manhattan_dist(neigh, endNode)
            
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    closedSet = {}
    openSet = {}
    cameFrom = {}

    gScore = {} #distance from start to current node
    fScore = {} #distance from start to current node + man dist of current node to end node

    startNode = maze.getStart()
    print("start node: {}".format(startNode))
    #print("startNode")
    #print(startNode)
    goalNodes = maze.getObjectives()
    endNode = None

    if (len(goalNodes) > 1):  
        path = []
        dots = create_dict_fromlist(goalNodes)
        print("dots: {}".format(dots))
        current_dot = get_closet_dot(startNode, dots)
        gScore[startNode] = 0 
        fScore[startNode] = manhattan_dist(startNode, current_dot)
        num_states_explored = 1

        openSet[startNode] = gScore[startNode] + manhattan_dist(startNode, current_dot)
        while (len(dots) != 0):
            while (len(openSet.keys()) != 0) :
                current = get_min_f_score(openSet, fScore)
                print("current: {}".format(current))
                print("fScore: {}".format(openSet))
                if (current == current_dot):
                    dots.pop(current_dot)
                    current_dot = get_closet_dot(current, dots)
                    #path.extend(reconstruct_path(cameFrom, current))
                    print("path: {}".format(path))
                    print("current dot: {}".format(current_dot))
                    print("dots: {}".format(dots))
                    break
                #print("openset")
                #print(openSet)
                #print("current")
                #print(current)
                closedSet[current] = openSet.pop(current)
                for neigh in maze.getNeighbors(current[0], current[1]):
                    if (maze.isValidMove(neigh[0], neigh[1])):
                        if (neigh in closedSet): #we already explored this node
                            continue
                        else:
                            tScore = gScore[current] + 1 #need help on finding distance between two points
                            if (neigh not in gScore):
                                gScore[neigh] = math.inf

                            if (neigh not in openSet):
                                openSet[neigh] = gScore[neigh] + manhattan_dist(neigh, current_dot)
                                num_states_explored += 1
                            elif (tScore >= gScore[neigh]):
                                continue
                            
                            gScore[neigh] = tScore
                            cameFrom[neigh] = current
                            fScore[neigh] = gScore[neigh] + manhattan_dist(neigh, current_dot)

            path.extend(reconstruct_path(cameFrom, current))
            #print("len: {} len2: {}".format(len(openSet), len(dots)))
        return path, num_states_explored
    else:
        endNode = goalNodes[0]    
        gScore[startNode] = 0 
        fScore[startNode] = manhattan_dist(startNode, endNode)
        num_states_explored = 1

        openSet[startNode] = gScore[startNode] + manhattan_dist(startNode, endNode)

        while (len(openSet.keys()) != 0):
            current = get_min_f_score(openSet, fScore)
            if (current == endNode):
                return reconstruct_path(cameFrom, current), num_states_explored
            
            #print("openset")
            #print(openSet)
            #print("current")
            #print(current)
            closedSet[current] = openSet.pop(current)

            for neigh in maze.getNeighbors(current[0], current[1]):
                if (maze.isValidMove(neigh[0], neigh[1])):
                    if (neigh in closedSet): #we already explored this node
                        continue
                    else:
                        tScore = gScore[current] + 1 #need help on finding distance between two points
                        
                        if (neigh not in gScore):
                            gScore[neigh] = math.inf

                        if (neigh not in openSet):
                            openSet[neigh] = gScore[neigh] + manhattan_dist(neigh, endNode)
                            num_states_explored += 1
                        elif (tScore >= gScore[neigh]):
                            continue
                        
                        gScore[neigh] = tScore
                        cameFrom[neigh] = current
                        fScore[neigh] = gScore[neigh] + manhattan_dist(neigh, endNode)
            
    return [], 0


def create_dict_fromlist(goalNodes):
    dots = {}
    for goal in goalNodes:
        dots[goal] = 1
    
    return dots

def get_closet_dot(current, dots):
    minDist = math.inf
    r_dot = None
    for dot in dots.keys():
        dist = manhattan_dist(current, dot)
        if (dist < minDist):
            minDist = dist
            r_dot = dot

    return r_dot

def get_path_length(came_from, current):
    path = reconstruct_path(came_from, current)
    #print("path length")
    #print(len(path))
    #print(path)
    return len(path)

def reconstruct_path(came_from, current):
    total_path = [current]
    while (current in came_from.keys()):
        current = came_from[current]
        total_path.append(current)
    
    #rint(total_path)
    return total_path

def get_min_f_score(openSet, fScore):
    #print("fScore: {}".format(fScore))
    #print("openSet: {}".format(openSet))
    minScore = math.inf
    minK = None
    for k in openSet.keys():
        minK = k

    for k in openSet.keys():
        #print("K: {}".format(k))
        if fScore[k] < minScore:
            minScore = fScore[k]
            #print(minScore)
            minK = k
    return minK

def get_distance(node1, node2):
    eudistance = math.sqrt(math.pow(node1[0]-node2[0],2) + math.pow(node1[1]-node2[1],2) )
    return eudistance

def manhattan_dist(startNode, endNode):
    totalSum = 0
    for i in range(0,2):
        totalSum += abs(startNode[i] - endNode[i])

    #("man dists: %d" %  (totalSum))
    return totalSum