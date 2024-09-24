# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm..

## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.

## POLICY ITERATION ALGORITHM
Environment Setup:

The code begins by importing necessary libraries and setting up the Frozen Lake environment using Gym. It also initializes the initial state, goal state, and transition probabilities (P).
Value Iteration Algorithm:

The core algorithm used in this code is Value Iteration. Value Initialization:
Initialize a value function (V) with zeros for each state. Value Iteration Loop:
For each state (s), calculate the Q-values for all possible actions (a) using the Bellman equation. The Q-value represents the expected cumulative reward when taking action 'a' from state 's'.
Update the value function (V) by taking the maximum Q-value for each state.
Check for convergence: If the maximum change in value function (V) is smaller than a threshold (theta), break the loop.
After convergence, derive the optimal policy (pi) by selecting actions that maximize the Q-values. Results:
The code prints the optimal policy, its state-value function, the success rate of reaching the goal, and the mean undiscounted return of the optimal policy.

## VALUE ITERATION FUNCTION
```
Developed By:P.Siva Naga Nithin
Reg.No:212221240037
```
```
desc=['FFSH','FFFH','GFHF','HFFH']
env = gym.make('FrozenLake-v1',desc=desc)
init_state = env.reset()
goal_state = 8
P = env.env.P
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/70d15615-2bc9-4135-a973-1175ad045513)

![image](https://github.com/user-attachments/assets/08b3e7a5-73d4-46e8-9d2f-53552e83b730)

![image](https://github.com/user-attachments/assets/f28891cf-26ac-4a31-97d5-f60221752b03)

## RESULT:

Thus, an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm is successfully implemented.
