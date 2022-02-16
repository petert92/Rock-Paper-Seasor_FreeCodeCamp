# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

#@title Sources
# https://colab.research.google.com/drive/1IlrlS3bB8t1Gd5Pogol4MIwUxlAjhWOQ#forceEdit=true&sandboxMode=true&scrollTo=jFRtn5dUu5ZI
# https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/amp/

#@title Imports
import numpy as np
from random import randint
from icecream import ic

#@title Play result
# Retunr reward depending on result between play
# input: player play, opponent play, rewards [int list]
# output: reward [int]
def playResult(myPlay, opponentPlay, rewards):
  if isinstance(myPlay, (int, np.int64)): myPlay = state2play[myPlay]
  if isinstance(opponentPlay, int): opponentPlay = state2play[opponentPlay]

  for ix, play in enumerate(state2play):
    if play == myPlay: # find my play in the ring
      if myPlay == opponentPlay: reward = rewards[1] # tie
      elif opponentPlay == state2play[ix-2]: reward = rewards[2] # loose
      else: reward = rewards[0] # win
      break
    
  return reward


#@title Update Q_matrix
# Return Q-matrix updated
# input: player play, lastState, opponent play, learning rate, gamma, rewards, Q-matrix
# output: same Q-matrix but updated
def update_Q_matrix(myPlay, lastState, opponentPlay, lr, gamma, rewards, Q_matrix):
  if isinstance(myPlay, str): myPlay = play2state[myPlay]
  if isinstance(lastState, str): lastState = play2state[lastState]
  if isinstance(opponentPlay, str): opponentPlay = play2state[opponentPlay]

  reward = playResult(myPlay, opponentPlay, rewards)

  updateValue = np.float64(Q_matrix[lastPlay, myPlay] + \
      lr * (reward + gamma * np.max(Q_matrix[lastState, :]) - Q_matrix[lastPlay, myPlay]))
  
  Q_matrix[lastState, myPlay] = updateValue
  return Q_matrix


#@title Main
def player(prev_play, opponent_history=[]):
  opponent_history.append(prev_play)

  # Using thwe test module opponent history doesn't start from cero
  # so I dischard the 1st n*1k datapoints when change the opponent
  if len(opponent_history) > 1000:
    aux = len(opponent_history) % 1000
    opponent_history = opponent_history[-aux:]

  # Hyperparameters
  learnigRate = 0.7
  gamma = 0.8
  rewards = [4, -3.5, -3.5]
  n_startUsingModel = 3
  randomConstant = 0.8
  randomDecrease = 0.05

  # Long time variables
  global Q_matrix
  global randomConst
  global lastPlay
  global lastState
  global state2play
  state2play = ['R','P','S']
  global play2state
  play2state = {myPlay:ix for ix,myPlay in enumerate(state2play)}
  firstModelPlay = True

  play = "R"
  # Random
  if 1 < (len(opponent_history)-1) < n_startUsingModel: # First 49 plays -> random 
    play = opponent_history[-1]
    lastState = play
  
  # Model plays
  elif len(opponent_history)-1 >= n_startUsingModel:

    if len(opponent_history)-1 > n_startUsingModel: firstModelPlay = False

    if firstModelPlay: 
      Q_matrix = np.zeros((len(state2play), len(state2play)), dtype=np.float64)
      randomConst = randomConstant
    else: Q_matrix = update_Q_matrix(lastPlay, lastState, opponent_history[-1], learnigRate, gamma, rewards, Q_matrix)

    # Random play so the agent do exploration
    if np.random.uniform(0, 1) < randomConst or firstModelPlay:
      play = randint(0,2)
      randomConst -= randomDecrease
      if not firstModelPlay: lastState = lastPlay
      lastPlay = play
      play = state2play[play]
    # Decision play
    else: 
      play = np.argmax(Q_matrix[lastPlay,:]) 
      lastState = lastPlay
      lastPlay = play
      play = state2play[play]
  
  # Using thwe test module must restart global variables when change the oppoent
  if (len(opponent_history)-1) == 999: 
    del Q_matrix, randomConst, lastPlay, lastState, state2play, play2state
    
  if isinstance(play, int): play = state2play[play]
  return play