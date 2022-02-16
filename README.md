# Rock-Paper-Seasor_FreeCodeCamp
Q-learning model to win Rock-Paper-Seasor game

To accomplish the ML course from https://www.freecodecamp.org/learn I must build 4 prjects. One of them is a model to win the game Rock-Paper-Seasor (at least 600/1000 (60%) of plays) to 4 diffferent algorithms. Full explenation in https://replit.com/@PedroTealdi/boilerplate-rock-paper-scissors-3#README.md

The model is a reinforcement type, just a simple Q Learning. The hyperparameters to set are:
- Learning rate (as common model)
- Gamma (as common model)
- Rewards: win, tie, loose
- Random constant: so the model explore "the enviorament"
- Random decrese: for each random play the random constant is decresed
- n plays to start using the model: as the datapoints are the plays themself, the agent can't start playing from the beggining 

Develope 2 functions:
- update_Q_matrix: compare the last plays from the agent and the opponent and update the Q-matrix
- playResult: return the reward depending on the play result

My main sources:
https://colab.research.google.com/drive/1IlrlS3bB8t1Gd5Pogol4MIwUxlAjhWOQ#forceEdit=true&sandboxMode=true&scrollTo=jFRtn5dUu5ZI
https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/amp/
