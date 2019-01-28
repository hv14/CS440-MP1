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
    return [], 0


def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    closedSet = {}
    openSet = {}
    cameFrom = {}

    gScore = {key: value for key, value in }

    startNode = maze.getStart()
    endNode = maze.getObjectives()
    openSet[startNode] = manhattan_dist(startNode, endNode)


    return [], 0

def manhattan_dist(startNode, endNode):
    totalSum = 0
    for i in range(0,2):
        totalSum += abs(startNode[i] - endNode[i])

    return totalSum