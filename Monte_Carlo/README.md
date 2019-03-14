## Monte Carlo Learning

Code for Monte Carlo Learning Algorithm.

## Algorithm to Calculate Returns:

1. Initialize G to 0
2. states_and_returns = []
3. Loop backward through the list of states_and_rewards(s,r):
    + appends(s,G ) to states_and_returns
    + G = r + gamma * G
4. reverse states_and_returns to the original order.

## Epsilons Greedy Algorithms

1. Generate a random number p, between 0 and 1
2. if p < (1 - ε) take the action dictated by policy.
3. otherwise take a random action.

## Monte Carlo Q Learning Algorithm

1. Initialize a policy to a random action for every state.
2. Initialize Q[(s,a)] to 0 for every possible state and action.
3. Initialize returns[s,a] as an empty array for each possible state and action.
4. Loop N times (anough for values to converge)
    + play the game and get a list of state_actions_returns
    + for(s, A, G) in state actions returns:
        + If we haven't seen this (s,a) pair so far in the game
            + append G to return[s,a]
            + Q[s][a] = mean(returns[s,a])
        + For each non-terminal state s:
            + Policy[s] = the action with the highest Q value for state s
5. for each state s, V[s] = Highest Q value for state s
6. return V policy

## Rewards

![](https://i.imgur.com/8JohinV.jpg)

## Output

![](https://i.imgur.com/1G6o8lG.jpg?1)

