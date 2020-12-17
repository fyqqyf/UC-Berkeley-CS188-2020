# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        #print("new_food:", newFood)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()#list
        ghostPos = []
        for G in newGhostStates:
            ghostPos_ = G.getPosition()[0], G. getPosition()[1]
            ghostPos.append(ghostPos_)
        #ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = newScaredTimes[0] > 0
        # if not new ScaredTimes new state is ghost: return lowest value
        if not scared and (newPos in ghostPos):
            return -2

        if scared and (newPos in ghostPos):
            return 1
        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)

        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1 / fd(closestFoodDist[0]) - 1/gd(closestGhostDist[0])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        PacmanValue = -100000.0
        PacmanAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)

            if nextValue > PacmanValue:
                PacmanValue = nextValue
                PacmanAction = action
        return PacmanAction



    def getValue(self, gameState, currentDepth, agentIndex):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:   #pacman的回合
            return self.PacmanValue(gameState,currentDepth)

        else:   #ghost的回合
            return self.GhostValue(gameState,currentDepth,agentIndex)



    def PacmanValue(self, gameState, currentDepth):
        PacmanValue = -100000.0
        for action in gameState.getLegalActions(0):
            PacmanValue = max(PacmanValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return PacmanValue

    def GhostValue(self, gameState, currentDepth, agentIndex):
        GhostValue = 100000.0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                GhostValue = min(GhostValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0))
            else:
                GhostValue = min(GhostValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))
        return GhostValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.getMax(gameState)[1]
        # util.raiseNotDefined()
    
    def getMax(self,gameState,depth=0,agentIndex=0,alpha=-100000.0,beta=100000.0):
        legalActions = gameState.getLegalActions(agentIndex)

        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState),None
            
        maxVal=None
        bestAction=None
        for action in legalActions:
            value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)[0]
            if value is not None and (maxVal == None or value > maxVal):
                maxVal = value
                bestAction = action
            if value is not None and value > beta:
                return value, action
            if value is not None and value > alpha:
                alpha = value
        return maxVal, bestAction

    def getMin(self, gameState, depth = 0, agentIndex = 0, alpha = -1000000.0,
               beta = 100000):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        minVal = None
        bestAction = None
        for action in legalActions:
            if agentIndex >= gameState.getNumAgents() - 1:
                value = self.getMax(gameState.generateSuccessor(agentIndex, action), depth+1, 0, alpha, beta)[0]
            else:
                value = self.getMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta)[0]
            if value is not None and (minVal == None or value < minVal):
                minVal = value
                bestAction = action
            if value is not None and value < alpha:
                return value, action
            if value is not None and value < beta:
                beta = value
        return minVal, bestAction
        



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.getMax(gameState)
    def getMax(self, gameState, depth = 0, agentIndex = 0):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        maxVal = None
        bestAction = None
        for action in legalActions:
            value = self.getExpectation(gameState.generateSuccessor(agentIndex, action), depth, 1)
            if value is not None and (maxVal == None or value > maxVal):
                maxVal = value
                bestAction = action
        if depth is 0 and agentIndex is 0:
            return bestAction
        else:
            return maxVal
    
    def getExpectation(self, gameState, depth = 0, agentIndex = 0):
        legalActions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState) 
        totalUtil = 0
        numActions = len(legalActions)
        for action in legalActions:
            if agentIndex >= gameState.getNumAgents() - 1:
                totalUtil += self.getMax(gameState.generateSuccessor(agentIndex, action), depth+1, 0)
            else:
                totalUtil += self.getExpectation(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1)
        return totalUtil / float(numActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        
    if len(GhostStates)>0:
        nearestGhost = min([manhattanDistance(Pos,ghostState.configuration.pos) for ghostState in GhostStates])
        if len(Food.asList()) > 0:
            if nearestGhost < 2:
                dangerScore = -10/float(nearestGhost+1)
            else:
                dangerScore=0
        else:
            dangerScore = 10000 / float(nearestGhost**2)
    else:
        dangerScore = 0
    
    if len(Food.asList()) > 0:
        nearestFood = (min([manhattanDistance(Pos, food) for food in Food.asList()]))
        nearFoodHeuristic = 1/float(nearestFood**2)
    else:
        nearFoodHeuristic = 0
    totalScaredTimes = sum(ScaredTimes)
    
    return currentGameState.getScore()*10+nearFoodHeuristic*2+dangerScore+totalScaredTimes/10

# Abbreviation
better = betterEvaluationFunction
