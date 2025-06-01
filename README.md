# EX_03 - POLICY ITERATION ALGORITHM

## AIM

To implement the Policy Iteration algorithm using Python in the FrozenLake-v1 environment and evaluate the resulting optimal policy based on its success rate and mean return.

## PROBLEM STATEMENT

The FrozenLake environment is a benchmark reinforcement learning task where an agent must navigate a grid-based frozen surface to reach a goal state without falling into holes. The environment is stochastic, meaning that the agent's actions have uncertain outcomes due to slippery tiles.

The environment is modeled as a Markov Decision Process (MDP) with:

- States: Each cell in the grid.

- Actions: Move left, right, up, or down.

- Transition Probabilities: Due to the slippery nature, intended actions might not always be executed.

- Rewards: +1 for reaching the goal (G), 0 otherwise.

The goal is to determine the optimal policy—a mapping from states to actions—that maximizes the cumulative reward. This is done using the Policy Iteration algorithm, which iteratively evaluates and improves a policy until convergence.

## POLICY ITERATION ALGORITHM:

Policy Iteration is a classic method for solving MDPs and consists of two main steps: Policy Evaluation and Policy Improvement, repeated until the policy stabilizes.

### Steps of the Policy Iteration Algorithm:

![image](https://github.com/user-attachments/assets/9f731f3b-cfac-4400-989f-2a3a585465ad)


### Policy Improvement Function:
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def policy_improvement(V, P, gamma=1.0):
    def improved_policy(s):
        action_values = []
        for a in range(len(P[s])):
            q_sa = 0
            for prob, next_state, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * V[next_state] * (not done))
            action_values.append(q_sa)
        return np.argmax(action_values)
    return improved_policy

```

### Policy Iteration Function
```
DEVELOPED BY : NIRAUNJANA GAYATHRI G R
REGISTER NO. : 212222230096
```
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = lambda s: 2  # start with all RIGHT actions
    stable = False
    while not stable:
        V = policy_evaluation(pi, P, gamma, theta)
        new_pi = policy_improvement(V, P, gamma)
        stable = True
        for s in range(len(P)):
            if pi(s) != new_pi(s):
                stable = False
                break
        pi = new_pi
    return V, pi
     
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy


![image](https://github.com/user-attachments/assets/19cda82d-7fb0-4bbd-84e3-e1598d501317)


![image](https://github.com/user-attachments/assets/67a32898-6cae-4826-b5be-337cf3a2f4d7)


### 2. Policy, Value function and success rate for the Improved Policy


![image](https://github.com/user-attachments/assets/72c05d28-804d-49fa-9cc4-b58ffd7a95c2)

![image](https://github.com/user-attachments/assets/691ee637-3a8e-400b-a3be-9e3f7fc5e464)

![image](https://github.com/user-attachments/assets/dba8947e-8789-48dc-a493-c626073bcd92)


### 3. Policy, Value function and success rate after policy iteration


![image](https://github.com/user-attachments/assets/63e83532-6c13-4d8b-a21c-e764e79643f6)


![image](https://github.com/user-attachments/assets/8e2226f8-da56-41da-8638-2ec4a2d808e4)


![image](https://github.com/user-attachments/assets/428e309a-e335-4ab2-b3b0-8fa3a87469f5)

## RESULT:

Thus to implement the Policy Iteration algorithm using Python in the FrozenLake-v1 environment and evaluate the resulting optimal policy based on its success rate and mean return is successfully implemented.
