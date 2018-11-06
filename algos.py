import os
from collections import defaultdict, Counter
import random
import setup
import copy

class SearchAlgo():
    # defining an abstraction of a search algorithm. Specific algorithms should have the functions below

    # function to return the score of the word based on a defined heurestic. Takes in the word to be evaluated,
    def getHeuristic(self, ch, count): raise NotImplementedError("Override me")

    def getWordToPlay(util, tiles, seed): raise NotImplementedError("Override me")

    # Since different styles of algorithm will have different simulation steps we define a sepsrate simulate for each algos.
    # Ideal would be to define a simulate that is common across algos but can be hard specially with different styles of algo.
    def Simulate(self): raise NotImplementedError("Override me")

    ## Functions common across all algos
    def checkForWord(self, util, soFar):
        base = ''
        for i in soFar:
            if i != '-': base += i
        base = base.upper()
        baseSorted = ''.join(sorted(base))
        if util.anagramMap[baseSorted]:
            return util.anagramMap[baseSorted][0]
        return ""

    def checkLeftRight(self, board, row, col, spacing):
        for dRow in range(-1,2):
            for dCol in range(-spacing, spacing + 1):
                currCol = col + dCol
                currRow = row + dRow
                try:
                    if board[currRow][currCol] != ' ' and currCol != col:
                        return False
                except IndexError:
                    return False
        return True

    def checkUpDown(self, board, row, col, spacing):
        for dCol in range(-1,2):
            for dRow in range(-spacing, spacing + 1):
                currCol = col + dCol
                currRow = row + dRow
                try:
                    if board[currRow][currCol] != ' ' and currRow != row:
                        return False
                except IndexError:
                    return False
        return True

    # The function below checks for each playable spot on the board,
    def getSpots(self, board, spacing):
        spots = []
        for r in range(setup.numRows(board)):
            for c in range(setup.numCols(board)):
                if board[r][c] != '-':
                    if self.checkLeftRight(board, r, c, spacing):
                        found = self.Spots(r, c, "left_right")
                        spots.append(found)
                    if self.checkUpDown(board, r, c, spacing):
                        found = self.Spots(r, c, "up_down")
                        spots.append(found)
        return spots

class BFS(SearchAlgo):

    def __init__(self, max_depth, Spot):
        self.max_depth = max_depth
        self.Spots = Spot

    def getWordToPlay(self, util, tiles, seed):
        best = ''
        word = ''
        bestscore = 0
        pipeline = []
        pipeline.append(seed)
        depth = 0
        while (not setup.isEmpty(pipeline)):
            soFar = pipeline.pop(0)
            score = self.getScore(util, soFar, seed)
            if score > bestscore:
                word = self.checkForWord(util, soFar)
            if word != '' and word != best:
                bestscore = score
                best = word
                print('The new best word is: {}'.format(best))

            idx = len(soFar) - len(seed)
            if idx < len(tiles):
                a = soFar + tiles[idx]
                pipeline.append(a)
                b = soFar + '-'
                pipeline.append(b)

            depth += 1

            if depth == self.max_depth: break

        return best

    def playWordOnBoard(self, util, tiles, board):
        bestSpot = (None, None, None)
        best = ''
        bestScore = 0
        playableSpots = self.getSpots(board, len(tiles))

        for spot in playableSpots:
            seed = board[spot.r][spot.c]
            word = self.getWordToPlay(util, tiles, seed)
            if word != '':
                score = self.getScore(util, word, seed)
                if score > bestScore:
                    bestSpot = spot
                    bestScore = score
                    best = word

        if best != '':
            seed = board[bestSpot.r][bestSpot.c]
            used = list(best)
            used.remove(seed)
            used = "".join(used)
            setup.removeTiles(tiles, used)
            setup.orientWord(board, bestSpot, best)
        return best

    ## Returns the score corresponding to word
    def getScore(self, util,soFar,seed):
        score = 0
        for i in range(len(seed), len(soFar), 1):
            if soFar[i] != '-':
                ch = soFar[i]
                score += util.letterScores[ch]
        return score

    # It is called only once during the setup to save scores corresponding to each alphabet basis the
    # frequency of occurrence.
    def getHeuristic(self, ch, count):
        '''
        :param ch: alphabet for which score is to be determined
        :param count: count of those alphabets int he game
        :return: heurestic score value
        '''
        if ch in ['a', 'e', 'i', 'u']:
            return 1
        if count <= 2: return 40
        if count <= 4: return 10
        if count <= 9: return 5

        return 3


class longest_word(SearchAlgo):

    def __init__(self, max_depth, Spot):
        self.max_depth = max_depth
        self.Spots = Spot

    def getWordToPlay(self, util, tiles, seed):
        if tiles == []: return ''

        # Start with anagram map, look for matches
        tiles_copy = copy.deepcopy(tiles)
        if seed != "" and seed != " ":
           tiles_copy.append(seed)
        tiles_copy = sorted(tiles_copy)

        # Loop through map, starting with longest possible length
        for i in range(len(tiles_copy),1,-1):
            for anagram in util.anagramMapCounts[i]:
                anagram_list = list(anagram)

                #Check for seed, which must be present in match
                if seed != "" and seed != " ":
                    if seed not in anagram_list:
                        continue
                
                # Check for perfect match
                if tiles_copy == anagram_list:
                    return util.anagramMap[anagram][0]

                # Check for partial match
                tiles_counter = Counter(tiles_copy)
                tiles_counter.subtract(Counter(anagram_list))
                if sum(1 for k,v in dict(tiles_counter).items() if v < 0) == 0:
                    intersect = list((Counter(tiles_copy) & Counter(anagram_list)).elements())
                    match = ''.join(sorted(intersect))
                    return util.anagramMap[match][0]

        return ''

    def getHeuristic(self, ch, count):
        pass

    def playWordOnBoard(self, util, tiles, board):
        playableSpots = self.getSpots(board, len(tiles))

        for spot in playableSpots:
            seed = board[spot.r][spot.c]
            if seed != '' and seed != ' ':
                word = self.getWordToPlay(util, tiles, seed)
                if word != '': 
                    break

        if word != '':      
            used = list(word)
            used.remove(seed)
            used = "".join(used)
            setup.removeTiles(tiles, used)
            setup.orientWord(board, spot, word)
        return word
