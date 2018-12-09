import numpy as np
import os
from collections import defaultdict
import random
import algos
from setup import setUpUtils
import setup
import time
import pickle

## Global variables
# How many tiles each player draws at the start
NUM_START_TILES = 23
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

def run_solver(algorithm, playertiles, begin=True):

    ## Initialize the algorithm object
    algo = algorithm(MAX_WORDS_PER_SPOT, Spot)
    setup.loadLetterScores(util, algo.getHeuristic)
    if begin:
    #1. Play the first word, starting with no words on the board
        first = getFirstWord(algo, util, playertiles)
        feature_vector['len_max_word'] = len(first)/NUM_START_TILES
        #print ("The first word is:",first)
        placeFirstWordOnBoard(board,first)
        setup.outputTrimmedBoard(board)
        setup.removeTiles(playertiles,first)
    #print ("remaining tiles : {}".format(playertiles))

    #2. Play remaining words
    # The algorithm is deferred to the algorithm class in algos.py file.
    while True:
        if algo.playWordOnBoard(util, playertiles, board) == "": break
    
    # trimmedBoard = setup.outputTrimmedBoard(board)
    # for row in trimmedBoard: print (*row)
    #print ("remaining tiles : {}".format(playertiles))

    return (board, playertiles)

def getAction(weights, state):
    score_append, score_reconstruct = 0, 0
    for k,v in state.items():
        score_append += v*weights['append'][k]
        score_reconstruct += v * weights['reconstruct'][k]

    if score_append >= score_reconstruct: return 'append'
    else: return 'reconstruct'

def testFunction():
    '''
    Function to run during the test time to evaluate the performance of the weights learned by Q-learning during the
    training phase.
    :return:
    '''
    def append():
        '''
        For the test function, try to append the tiles in hand on the existing crossword
        :return:
        '''
        s = time.time()
        (board, playertiles) = run_solver(algos.longest_word, replay_tiles, begin=False)
        e = time.time()
        time_taken = e - s
        if playertiles == []:
            return completetion_reward - time_penalty * time_taken
        else: return 0

    def reconstruct():
        '''
        For the test function reconstruct the entire crossword, given the additional set of tiles.
        :return:
        '''
        #print ("indes {}, length of the list {}".format(i,len(random_tracker)))
        recon_tiles = randomtiles.copy() + newtiles
        s = time.time()
        (board, playertiles) = run_solver(algos.longest_word,recon_tiles)
        e = time.time()
        time_taken = e - s
        if playertiles == []:
            return completetion_reward - time_penalty * time_taken
        else: return 0
    toss = random.random()
    print ("random is : ",toss)
    if  toss >= 0.5: random_r = append()
    else: random_r = reconstruct()
    if random_r !=0:
        random_tracker[i] = 1
        print("completed in random policy")

    # learnt policy
    action =  getAction(learned_weights,feature_vector)
    print("action by optimal policy is : " + action)
    if action == 'append': policy = append()
    else: policy = reconstruct()
    if policy != 0:
        policy_tracker[i] = 1
        print("completed in learned policy")
    return random_r, policy

if __name__ == '__main__':

    train = False
    run_alg1 = False
    n_runs = 100 # no of iterations

    # Initialize lists for completion metrics
    alg1_time = []
    alg2_time = []
    alg1_complete = []
    alg2_complete = []

    # For MDP--initialize weights, rewards, and penalties
    Q_opt = defaultdict(lambda: defaultdict(float))
    completetion_reward = 100
    time_penalty = 1 #weight for time penaly
    eta = 1
    epsilon = 0.1 # gradient decent parameter
    no_new_tiles = 1
    policy = defaultdict(str)
    candidates = []
    # Read in dictionary
    util = Util()
    loadedTiles = list(setUpUtils(util))
    simulation_tracker = defaultdict(int)
    weights = {'append': defaultdict(float), 'reconstruct': defaultdict(float)}
    saved_weights = []
    # for testing
    policy_tracker = [0]*100
    random_tracker = [0]*100
    with open('saved_weights_1000_iter.pickle', 'rb') as f:
        mynewlist = pickle.load(f)
    learned_weights = mynewlist[-1]
    random_reward = []
    policy_reward = []
    for i in range(100):
        print ("Iteration no : ",i)
        ## Solve step 1 first
        feature_vector = {'no_vowels': 0, \
                          'no_BCHMP': 0, \
                          'no_DLN': 0, \
                          'no_FGJK': 0, \
                          'no_RSTN': 0, \
                          'no_QVWXYZ': 0, \
                          # features specific for board
                          # no_words = 0
                          'len_max_word': 0, \
                          # features specific for new tiles
                          'no_tiles_hand': 0,
                          }
        allTiles = loadedTiles.copy()
        (randomtiles, remainingPile) = setup.selectRandomTiles(allTiles, NUM_START_TILES)
        board = setup.makeBoard(MAX_BOARD_SIZE)
        begin_tiles = randomtiles.copy()
        start = time.time()
        (board, unplayed_tiles) = run_solver(algos.longest_word,begin_tiles)
        end = time.time()
        time_taken = end - start

        if unplayed_tiles == []: alg2_complete.append(1)
        else: alg2_complete.append(0)
        alg2_time.append(time_taken)

        # Start Step 2
        # Set up MDP
        # Append new letter to hand
        ## making the new feature vector. Features selected are
        for char in randomtiles:
            if char in 'AEIOU': feature_vector['no_vowels'] += 1 / NUM_START_TILES
            if char in 'BCHMP': feature_vector['no_BCHMP'] += 1 / NUM_START_TILES
            if char in 'DLN': feature_vector['no_DLN'] += 1 / NUM_START_TILES
            if char in 'FGJK': feature_vector['no_FGJK'] += 1 / NUM_START_TILES
            if char in 'RSTN': feature_vector['no_RSTN'] += 1 / NUM_START_TILES
            if char in 'QVWXYZ': feature_vector['no_QVWXYZ'] += 1 / NUM_START_TILES
        newtiles, remainingPile = setup.selectRandomTiles(remainingPile, no_new_tiles)
        replay_tiles = unplayed_tiles + newtiles #tiles after first peel call
        feature_vector['no_tiles_hand'] = 3*len(replay_tiles)/(len(replay_tiles) + len(randomtiles))
        new_tiles = ''.join(sorted(replay_tiles))
        feature_vector[''.join(sorted(replay_tiles))] = 1 # feature corresponding to tiles in hand
        ##TODO for greater than 2 tiles drawn we would want to have similar to above feature vectors for new tiles
        #MDP_state = ''.join(sorted(playertiles))
        MDP_state = feature_vector

        if 1==0:
            r_reward, p_reward = testFunction()
            random_reward.append(r_reward)
            policy_reward.append(p_reward)

        if train:
            print("runnning iteration : {}".format(i))
            if (i)%50 == 0:
                saved_weights.append(weights)
                print("saved weights")
                with open('saved_weights_1000_iter.pickle', 'wb') as handle:
                    pickle.dump(saved_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

            eta = 1 / (1 + simulation_tracker[new_tiles])
            simulation_tracker[new_tiles] = simulation_tracker[new_tiles] + 1

        #print('the original tiles are: {}'.format(randomtiles))
        #input('Press enter to run: ')
        #print("new hand:", playertiles)

            # Option 1: find place on current board
            #input('Press enter to run Option 1: ')
            Q_hat = 0
            for k,v in feature_vector.items():
                Q_hat += v*weights['append'][k]
            s = time.time()
            (board, playertiles) = run_solver(algos.longest_word, replay_tiles,begin = False)
            e = time.time()
            time_taken = e-s
            print ("Remaining tiles after appending is {}".format(playertiles))
            print ("the time taken for state {} for \n appending is: {}".format(MDP_state,time_taken))
            if playertiles == []:
                utility = completetion_reward - time_penalty*time_taken
            else:
                utility = 0
            # Q_opt[MDP_state]['append'] = (1 - eta) * Q_opt[MDP_state]['append'] + eta * utility
            for k,v in feature_vector.items():
                #if k != replay_tiles:
                weights['append'][k] = weights['append'][k] - epsilon * (Q_hat - utility) * v
                #else: weights['append'][k] = weights['append'][k] - eta * (Q_hat - utility) * v/7

            # Option 2: Break down the board and Reconstruct the whole board
            #input('Press enter to run Option 2: ')
            Q_hat = 0
            for k,v in feature_vector.items():
                Q_hat += v*weights['reconstruct'][k]
            playertiles = randomtiles.copy() + newtiles
            board = setup.makeBoard(MAX_BOARD_SIZE)
            s = time.time()
            (board, playertiles) = run_solver(algos.longest_word,playertiles)
            e = time.time()
            time_taken = e - s
            print ("Remaining tiles after reconstructing is {}".format(playertiles))
            print ("the time taken for state {} for \n reconstructing is: {}".format(MDP_state,time_taken))
            if playertiles == []:
                utility = completetion_reward - time_penalty * time_taken
            else:
                utility = 0
            # Q_opt[MDP_state]['reconstruct'] = (1 - eta) * Q_opt[MDP_state]['reconstruct'] + eta * utility
            for k, v in feature_vector.items():
                #if k != replay_tiles:
                weights['reconstruct'][k] = weights['reconstruct'][k] - epsilon * (Q_hat - utility) * v
                #else: weights['reconstruct'][k] = weights['reconstruct'][k] - eta * (Q_hat - utility) * v/7
            #7 here is number of features. normalizing in some sense so that the estimate doesn't blow up

    print ("policy reward is {}".format(policy_reward))
    print ("random reward is {}".format(random_reward))




# if run_alg1:
#     start = time.time()
#     board = setup.makeBoard(MAX_BOARD_SIZE)
#     playertiles = randomtiles.copy()
#     board, playertiles = run_solver(algos.BFS)
#     end = time.time()
#     if playertiles == []:
#         alg1_complete.append(1)
#     else:
#         alg1_complete.append(0)
#     alg1_time.append(end - start)
#
#  for key in Q_opt:
# #     if Q_opt[key]['append'] >= Q_opt[key]['reconstruct']: policy[key] = 'append'
# #     else: policy[key] = 'reconstruct'
#
# if run_alg1:
#     print("Alg1 runtime:", sum(alg1_time) / n_runs)
#     print("Alg1 complete:", sum(alg1_complete) / n_runs)
#
# print("Alg2 runtime:", sum(alg2_time) / n_runs)
# print("Alg2 complete:", float(sum(alg2_complete) / n_runs))
# print("Q_opt:", Q_opt)
# print ("optimal policy for all states is {}".format(policy))
# print("States explored and their frequency in simulaiton is : {}".format(simulation_tracker))

