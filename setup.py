import os
from collections import defaultdict
import random

###############################################################################################
#                                   Setup
###############################################################################################

'''
/* Make Board
 * Make a really large empty board. Make it so big that if we start in
 * the middle no combination of words could take us off the board.
 * Later when visualizing the board we can trim it.
 */
'''
def makeBoard(max_size):
    grid = []
    for _ in range(max_size):
        grid.append([])
        for _ in range(max_size):
            grid[-1].append(' ')
    return grid

'''
/* Set Up Utils
 * This function loads the scrabble dictionary, the map
 * of letter scores (the "scores" are made up by Chris and
 * are not part of the game rules) and the game tiles.
 */
'''
def setUpUtils(utils): # why is passed by reference here. Needs to understand this to know what name to put
    #print("Bananagrams")
    # with open("Collins-Scrabble-Words-2015.txt") as word_file:
    #     valid_words = set(word_file.read().split())
    vocab = set()
    with open("Collins-Scrabble-Words-2015.txt") as f:
        for line in f.readlines()[2:]:
            line = line.rstrip("\n")
            vocab.add(line)
    populateAnagramMap(vocab, utils)
    return loadLetters()

'''
For every word we create a map from it's sorted letters to the word. 
Basically this dictionary is a look up for anagrams
'''
def populateAnagramMap(eng_dict,utils):
    for word in eng_dict:
        word = word.upper()
        sorted_words = ''.join(sorted(word)) #sorted function sorts alphabetically and returns a list
        if utils.anagramMap[sorted_words]:
            utils.anagramMap[sorted_words].append(word)
        else:
            utils.anagramMap[sorted_words] = [word]

    # Create dictionary of word lengths
    for k in utils.anagramMap:
        length = len(k)
        utils.anagramMapCounts[length].append(k)
    return

'''
/* Load Letters
 * Bananagrams uses a carefully chosen count of tiles. I looked
 * up those counts and put them in the file banana-dist.txt. This
 * function reads that file and returns all the tiles as one long
 * string.
 */
'''
def loadLetterScores(utils, scoring):
    with open("bananagram-tiles.txt") as f:
        for line in f.readlines():
            ch,count = line.split(':')[0], int(line.split(':')[1])
            letterscore = scoring(ch,count)
            utils.letterScores[ch] = letterscore
    return

def loadLetters():
    with open("bananagram-tiles.txt") as f:
        letters = ""
        for line in f.readlines():
            ch,count = line.split(':')[0], int(line.split(':')[1])
            letters += ch*count
    #print("loaded letters are: {}".format(letters))
    return letters


def selectRandomTiles(pile, num):
    '''
    Given a pile of tiles randomly select the number without replacement. Return the selected tiles
    '''
    hand = []
    for i in range(num):
        idx = random.randint(0,len(pile) - 1)
        tile = pile[idx]
        del pile[idx]
        hand.append(tile)

    return hand, pile


###############################################################################################
#                                           Helper Functions
###############################################################################################
def removeTiles(playertiles, playedword):
    playedword = list(playedword.upper())
    for w in playedword:
        if w == '-' : continue
        playertiles.remove(w)
    return playertiles

def countLetters(string):
    count = sum([1 for char in string if char != '-'])
    return count

def orientWord(board, spot, word):
    # print(spot,word)
    seed = board[spot.r][spot.c]
    idx = word.index(seed)
    if spot.dir == "left_right":
        startCol = spot.c - idx
        for i,j in enumerate(word):
            board[spot.r][startCol + i] = j
    if spot.dir == "up_down":
        startRow = spot.r - idx
        for i,j in enumerate(word):
            board[startRow + i][spot.c] = j

    return


## Generic functions for using lists as numpy arrays
def isEmpty(l):
    return len(l) == 0

def numRows(l):
    return len(l)

def numCols(l):
    '''
    Assumes that each sub list has the same number of elements. Grid representation in lists essentially
    :param l: a list of lists
    :return: length of a sub list of the lists
    '''
    return len(l[0])

###############################################################################################
#                                           Board Visualization
###############################################################################################
def outputTrimmedBoard(board):
    minC = numCols(board) - 1
    maxC = 0
    minR = numRows(board) - 1
    maxR = 0
    for r in range(0, numRows(board)):
        for c in range(0, numCols(board)):
            if board[r][c] != ' ':
                minC = min(minC, c)
                minR = min(minR, r)
                maxC = max(maxC, c)
                maxR = max(maxR, r)
    newC = maxC - minC + 1
    newR = maxR - minR + 1
    trimmed = [[' ' for ii in range(newC)] for i in range(newR)]
    for r in range(newR):
        for c in range(newC):
            trimmed[r][c] = board[minR + r][minC + c]
    return trimmed
