import gym
import numpy as np
import time

#executes an episode
def execute(env,policy,gamma=1.0, render=False):
    start = env.reset()
    totalReward=0
    stepIndex=0
    episodeLength =100
    for t in range(episodeLength):
        if render:
            env.render()
        start,reward,done, _ = env.step(int(policy[start]))
        totalReward+=(gamma**stepIndex*reward)
        stepIndex+=1
        if done:
            break
    return totalReward

#Extract the policy given a value-function
def evaluate_policy(env,policy,gamma= 1.0,n=1000):
    scores = [execute(env,policy,gamma, False) for _ in range(n)]
    return np.mean(scores)

#Iteratively calculates the value-function under policy.
def CalcPolicyValue(env,policy, gamma=1.0):
    value = np.zeros(env.env.nS)
    eps = 1e-10
    while True:
        previousValue = np.copy(value)
        for states in range(env.env.nS):
            policy_a= policy[states]
            value[states] = sum([p*(r+gamma * previousValue[s_]) for p,s_,r,_ in env.env.P[states][policy_a]])
        if (np.sum((np.fabs(previousValue-value)))<=eps):
                break
    return value

#Extract the policy given a value-function
def extractPolicy(v,gamma=1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.env.nA)
        for a in range(env.env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.env.P[s][a]])
        policy[s]= np.argmax(q_sa)
    return policy


def policyIteration(env,gamma=1.0):
    policy = np.random.choice(env.env.nA, size=(env.env.nS)) #Initialize a random policy
    max_iterations = 100
    for i in range(max_iterations):
        oldPolicyValue = CalcPolicyValue(env, policy, gamma)
        newPolicy = extractPolicy(oldPolicyValue, gamma)
        if np.all(policy == newPolicy):
            print("Policy iteration converged at %d" %(i+1))
            break
        policy  = newPolicy
    return policy
#Solve each Environment with different gamma values
def SolveEnv(env,methods, envName):
    print(f"Solving Environment {envName}")
    for method in methods:
        name, func , gamma = method
        startTime = time.time()
        optimalPolicy = policyIteration(env,gamma)
        endTime = time.time()
        print(f'It took {endTime - startTime} seconds to compute a policy using "{name}" with gamma={gamma}')
        scores = evaluate_policy(env, optimalPolicy, gamma = 1.0)
        print(f'Policy average reward is {scores}')

if __name__ =="__main__":
    env = gym.make('FrozenLake8x8-v0')
    methods = [
                ('Policy Iteration', policyIteration, 0.9),
                ('Policy Iteration', policyIteration, 0.98),
                ('Policy Iteration', policyIteration, 1),
                ]
    frozenlake4 = gym.make('FrozenLake-v0')
    SolveEnv(env,methods, 'Frozen Lake 4x4')
    frozenlake8 = gym.make('FrozenLake8x8-v0')
    SolveEnv(env,methods, 'Frozen Lake 8x8')
