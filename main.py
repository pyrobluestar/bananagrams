import numpy as np
import os
from collections import defaultdict
import random
import algos
from setup import setUpUtils
import setup
import time

## Global variables
# How many tiles each player draws at the start
NUM_START_TILES = 21
# How big to make the virtual grid that I populate
MAX_BOARD_SIZE = NUM_START_TILES * 2
# Bananagrams is a complex game. Limit how deep we go
MAX_WORDS_PER_SPOT = 50000

class Spot():
    def __init__(self,r,c,dir):
        '''
        :param r: the row of the spot
        :param c: column of the spot
        :param dir: direction to move
        '''
        self.r = r
        self.c = c
        self.dir = dir

class Util():
    def __init__(self):
        self.anagramMap = defaultdict(list)
        self.anagramMapCounts = defaultdict(list)
        self.letterScores = defaultdict(int)

###############################################################################################
#                                   FIRST WORD
###############################################################################################
'''
Get First word. Chose a first word to play given your tiles. All df the logic here is deferred to the function 
getWordToPlay which can take a letter to pay off. At start there is no letter.
'''

def getFirstWord(algo, util, tiles):
    return algo.getWordToPlay(util,tiles,"")

def placeFirstWordOnBoard(board, first):
    for i in range(len(first)):
        middle = int(MAX_BOARD_SIZE/2)
        col = middle + i
        board[middle][col] = first[i]

def run_solver(algorithm):

    ## Initialize the algorithm object
    algo = algorithm(MAX_WORDS_PER_SPOT, Spot)
    setup.loadLetterScores(util, algo.getHeuristic)
    #1. Play the first word
    first = getFirstWord(algo, util, playertiles)
    #print ("The first word is:",first)
    placeFirstWordOnBoard(board,first)
    setup.outputTrimmedBoard(board)
    setup.removeTiles(playertiles,first)
    #print ("remaining tiles : {}".format(playertiles))

    #2. Play remaining words
    while True:
        if algo.playWordOnBoard(util, playertiles, board) == "": break
    
    trimmedBoard = setup.outputTrimmedBoard(board)
    #for row in trimmedBoard: print (*row)
    #print ("remaining tiles : {}".format(playertiles))

    return (board, playertiles)

if __name__ == '__main__':

    run_alg1 = False
    n_runs = 10

    # Initialize lists for completion metrics
    alg1_time = []
    alg2_time = []
    alg1_complete = []
    alg2_complete = []

    # For MDP--initialize weights, rewards, and penalties
    Q_opt = defaultdict(lambda: float(0))
    reward_append = 4
    pen_append = -1
    reward_reconst = 1
    pen_reconst = -1
    eta = 0.1

    # Read in dictionary
    util = Util()
    loadedTiles = list(setUpUtils(util))

    for i in range(n_runs):

        allTiles = loadedTiles.copy()
        (randomtiles, remainingPile) = setup.selectRandomTiles(allTiles,NUM_START_TILES)
        #print('the original tiles are: {}'.format(randomtiles))
        #input('Press enter to run: ')

        if run_alg1:
            start = time.time()
            board = setup.makeBoard(MAX_BOARD_SIZE)
            playertiles = randomtiles.copy()
            board, playertiles = run_solver(algos.BFS)
            end = time.time()
            if playertiles == []: alg1_complete.append(1)
            else: alg1_complete.append(0)
            alg1_time.append(end - start)

        start = time.time()
        board = setup.makeBoard(MAX_BOARD_SIZE)
        playertiles = randomtiles.copy()
        (board, playertiles) = run_solver(algos.longest_word)
        end = time.time()
        if playertiles == []: alg2_complete.append(1)
        else: alg2_complete.append(0)
        alg2_time.append(end - start)

        # Set up MDP
        # Append new letter to hand
        newtiles, remainingPile = setup.selectRandomTiles(remainingPile, 1)
        playertiles += newtiles
        MDP_state = ''.join(sorted(playertiles))
        #print("new hand:", playertiles)

        # Option 1: find place on current board
        #input('Press enter to run Option 1: ')
        (board, playertiles) = run_solver(algos.longest_word)
        if playertiles == []: 
            Q_opt[(MDP_state,'append')] = (1 - eta) * Q_opt[(MDP_state,'append')] + eta * reward_append
        else: 
            Q_opt[(MDP_state,'append')] = (1 - eta) * Q_opt[(MDP_state,'append')] + eta * pen_append

        # Option 2: break down board and play from scratch
        #input('Press enter to run Option 2: ')
        playertiles = randomtiles.copy() + newtiles
        board = setup.makeBoard(MAX_BOARD_SIZE)
        (board, playertiles) = run_solver(algos.longest_word)
        if playertiles == []: 
            Q_opt[(MDP_state,'reconstruct')] = (1 - eta) * Q_opt[(MDP_state,'reconstruct')] + eta * reward_reconst
        else: 
            Q_opt[(MDP_state,'reconstruct')] = (1 - eta) * Q_opt[(MDP_state,'reconstruct')] + eta * pen_reconst


    if run_alg1:
        print("Alg1 runtime:", sum(alg1_time) / n_runs)
        print("Alg1 complete:", sum(alg1_complete) / n_runs)

    print("Alg2 runtime:", sum(alg2_time) / n_runs)
    print("Alg2 complete:", sum(alg2_complete) / n_runs)
    print("Q_opt:", Q_opt)

