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
from copy import deepcopy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)


def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_states_explored = 1
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

    #print(path)
    path.reverse()
    return path, num_states_explored


def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_states_explored = 1
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

    path.reverse()
    return path, num_states_explored


def greedy(maze):
    # return path, num_states_explored
    closedSet = {}
    openSet = {}
    cameFrom = {}

    gScore = {} #distance from start to current node
    fScore = {} #distance from start to current node + man dist of current node to end node

    startNode = maze.getStart()
    endNode = maze.getObjectives()[0]

    fScore[startNode] = manhattan_dist(startNode, endNode)
    num_states_explored = 1

    openSet[startNode] = manhattan_dist(startNode, endNode)

    while (len(openSet.keys()) != 0):
        current = get_min_f_score(openSet, fScore, gScore)
        if (current == endNode):
            return reconstruct_path(cameFrom, current), num_states_explored

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

# This function handles the multiple dots case.
def astar(maze):

        start = maze.getStart()
        goals = maze.getObjectives()
        #current_goal = greedy_get_best_goal_node(maze, start, goals)
        current_goal = optimal_get_best_goal_node(maze, start, goals)
        print("current goal: {}".format(current_goal))

        goals.remove(current_goal)

        path = []
        num_states_explored = 0

        path, num_states_explored = astar_driver(maze, start, current_goal)
        path.reverse()

        current_pos = deepcopy(current_goal)
        #print("Order of Goals:")

        while (len(goals) > 0):
            #current_goal = greedy_get_best_goal_node(maze, current_pos, goals)
            current_goal = optimal_get_best_goal_node(maze, current_pos, goals)
            print("current goal: {}".format(current_goal))

            goals.remove(current_goal)
            curr_path, curr_num_states_explored = astar_driver(maze, current_pos, current_goal)

            curr_path.reverse()
            path.extend(curr_path[1:])

            num_states_explored += curr_num_states_explored
            current_pos = deepcopy(current_goal)

            # Check if we have passed a goal node, if so remove it
            for node in path:
                for goal in goals:
                    if (node == goal):
                        goals.remove(node)

        #print("Final Path")
        #print(path)
        return path, num_states_explored


def astar_driver(maze, start, goal):
    # return path, num_states_explored
    closedSet = {} # explored
    openSet = {} # unexplored
    cameFrom = {} # predecessors

    gScore = {} #distance from start to current node
    fScore = {} #distance from start to current node + man dist of current node to end node

    startNode =  start
    endNode = goal

    gScore[startNode] = 0
    fScore[startNode] = manhattan_dist(startNode, endNode)
    num_states_explored = 1

    openSet[startNode] = gScore[startNode] + manhattan_dist(startNode, endNode)

    while (len(openSet.keys()) != 0):
        current = get_min_f_score(openSet, fScore, gScore)
        if (current == endNode):
            return reconstruct_path(cameFrom, current), num_states_explored

        closedSet[current] = openSet.pop(current)

        for neigh in maze.getNeighbors(current[0], current[1]):
            if (maze.isValidMove(neigh[0], neigh[1])):
                if (neigh in closedSet): #we already explored this node
                    if (gScore[current] + manhattan_dist(neigh, endNode) + 1 < fScore[neigh]):
                        print("did this even matter")
                        gScore[neigh] = gScore[current] + manhattan_dist(neigh, endNode) + 1
                        cameFrom[neigh] = current
                        fScore[neigh] = gScore[neigh] + manhattan_dist(neigh, endNode)
                        #continue
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

def calculate_MST(nodes):
    costs = {}
    predecessors = {}
    edge_costs = {}
    sum = 0

    # initialize dictionaries
    for node in nodes:
        costs[node] = math.inf
        predecessors[node] = None

    #arbritrarily pick a start node
    start = nodes[0]
    costs[start] = 0
    predecessors[start] = None

    MST_set = [start]

    for i in range(0, len(nodes)):
        current_node = nodes[i]
        MST_set.append(current_node)

        for neighbor in nodes:
            # Finding neighbors of current node that aren't in the MST_set
            if (neighbor != current_node and neighbor not in MST_set):
                if (manhattan_dist(current_node, neighbor) < costs[current_node]):
                    costs[neighbor] = manhattan_dist(current_node, neighbor)
                    predecessors[neighbor] = current_node

    for node, cost in costs.items():
        sum += cost


    return sum


def create_edge_list(maze, dots):
    edges = []
    weights = []
    for i in range(0, len(dots)):
        for j in range(0, len(dots)):
            if (i != j and i < j):
                edges.append((dots[i],dots[j]))
                weights.append(len(astar_driver(maze, dots[i], dots[j])[0]))

    return edges, weights

def create_set(dots):
    vertices = deepcopy(dots)
    for k in range(len(vertices)):
        vertices[k] = [vertices[k]]

    return vertices

def find_set(vertices, a):
    for i in range(0, len(vertices)):
        for j in range(0, len(vertices[i])):
            if (vertices[i][j] == a):
                return i

def union(vertices, a, b):
    i = find_set(vertices, a)
    j = find_set(vertices, b)
    for elem in vertices[j]:
        vertices[i].append(elem)
    vertices.pop(j)

def kruskal(edges, weights, dots):
    sEdges, sWeights = sort_edges(edges, weights)

    vertices_set = create_set(dots)
    count, i, sum = 0,0,0
    while len(vertices_set) > 1:
        if (find_set(vertices_set, sEdges[i][0]) != find_set(vertices_set, sEdges[i][1])):
            count += 1
            sum += sWeights[i]
            union(vertices_set, sEdges[i][0], sEdges[i][1])
        i+=1

    return sum


def sort_edges(edges, weights):
    if (len(edges) != len(weights)):
        return
    minWeight = math.inf
    minEdge = None

    finalEdges = []
    finalWeights = []

    lenW = len(weights)
    for x in range(0, lenW):
        minWeight = math.inf
        for i in range(0, lenW):
            if (weights[i] < minWeight and edges[i] not in finalEdges):
                minWeight = weights[i]
                minEdge = edges[i]
        finalEdges.append(minEdge)
        finalWeights.append(minWeight)

    return finalEdges, finalWeights


def create_dot_graph(maze, dots):
    adj_matrix = []
    for i in range(len(dots)):
        adj_matrix.append([])
        for j in range(len(dots)):
            adj_matrix[i].append(0)
    ids = {}
    z = 0
    for dot in dots:
        ids[dot] = z
        z += 1

    for x in dots:
        for y in dots:
            if (x > y):
                continue
            adj_matrix[ids[x]][ids[y]] = [x, y, manhattan_dist(x, y)]

    #print(ids)
    return adj_matrix
# Heuristic for optimally selecting the next goal node. Picks the one with the lowest
# summed MST edge weights.
def optimal_get_best_goal_node(maze, startNode, goals):
    if (len(goals) == 1):
        return goals[0]

    MST_costs = {}
    man_dists = {}
    for goal in goals:
        temp_goals = deepcopy(goals)
        temp_goals.remove(goal)
        edges, weights = create_edge_list(maze, temp_goals)
        MST_costs[goal] = kruskal(edges, weights, temp_goals)
        man_dists[goal] = math.inf
        for other in temp_goals:
            if (manhattan_dist(goal, other) < man_dists[goal]):
                man_dists[goal] = manhattan_dist(goal, other)

    sum = math.inf
    bestGoal = None
    for goal in goals:
        tSum = len(astar_driver(maze, startNode, goal)[0]) + MST_costs[goal] + man_dists[goal]
        if (tSum < sum):
            sum = tSum
            bestGoal = goal
        if (tSum == sum and bestGoal != goal):
            print("tie break: {} {}".format(goal, bestGoal))
            if (manhattan_dist(startNode, goal) < manhattan_dist(startNode, bestGoal)):
                bestGoal = goal
    return bestGoal

# heuristic for multiple dots problem, extra credit version
# returns the best goal from the startNode
def greedy_get_best_goal_node(maze, startNode, goals):

    curr_path, curr_num_nodes_expanded = astar_driver(maze, startNode, goals[0])
    curr_cost = len(curr_path)
    closest_goal = goals[0]

    #MST_costs = calculate_MST(maze, goals)

    for goal in goals:
        temp_path, temp_num_nodes_expanded = astar_driver(maze, startNode, goal)
        temp_cost = len(temp_path)

        if (temp_cost < curr_cost):
            closest_goal = goal
            curr_cost = temp_cost
            curr_num_nodes_expanded = temp_num_nodes_expanded

        # check if there is a tie between goal nodes
        if (temp_cost == curr_cost and goal != closest_goal ):
            if (temp_num_nodes_expanded < curr_num_nodes_expanded):
                closest_goal = goal
                curr_cost = temp_cost
                curr_num_nodes_expanded = temp_num_nodes_expanded

    return closest_goal

def chebyshev_dist(startNode, endNode):
    differences = []
    for i in range(0,2):
        differences.append(abs(startNode[i] - endNode[i]))
    if (differences[0] > differences[1]):
        return differences[0]
    else:
        return differences[1]

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

    #print(total_path)
    return total_path

def get_min_f_score(openSet, fScore, gScore):
    #print("fScore: {}".format(fScore))
    #print("openSet: {}".format(openSet))
    minScore = math.inf
    minK = math.inf
    current_min_node = None

    #print(fScore)
    #print(gScore)

    for k in openSet.keys():
        #print("K: {}".format(k))
        if fScore[k] < minScore:
            minScore = fScore[k]
            #print(minScore)
            minK = k
        if fScore[k] == minScore:
            if (gScore[k] > gScore[minK]):
                minScore = fScore[k]
                minK = k

    return minK

def get_distance(node1, node2):
    eudistance = math.sqrt(math.pow(node1[0]-node2[0],2) + math.pow(node1[1]-node2[1],2) )
    return eudistance

def manhattan_dist(startNode, endNode):
    totalSum = 0
    for i in range(0,2):
        totalSum += abs(startNode[i] - endNode[i])
    return totalSum
