consider a rover that has a fork in a path and it needs to decide what is the best route for it to take.
On one side there is a short path with a relatively low reward and the other side there is a longer path with a high reward and in between are many states (or distance from the reward).
#### Return function
The goal of the return function is to decide whether it's worth it or not to pursue one path or not. The formula goes as such:
```python
states = [100, 0, 0, 0, 0, 40]
# this seems botched thanks gpt
gamma = 0.9
def calculate_return(states, currState, gamma):
	newStates = []
	for enum, i in enumerate(states):
		
	return return_sum
```

### Policy
Find a policy "pi" that tells you what action (a = pi(s)) to take in every state (s) so as to maximise the return.

### Markov Decision Process (MDP)
The future only depends on where you are now and not where you came from.

### State action value function
Q(s, a) = return if you start in state s and take action a (once) then behave optimally after that

### Bellman Equation
- s: current state
- R(s): Reward of current state
- a: current action
- s': state you get to after taking action a
- a': action you take after s'
- Q is a linear regression function

```python
QofSA = R(s) + gamma * ((aMax / aPrime) * Q(sPrime, aPrime))
```

### Continuous state spaces
the simplified mars rover can only be in one of 6 position - but most robots a can be in 1 of and infinite amount of positions, that is an example of a continuous space

### Learning algorithm
```python
# initialize neural network with random weights and biases
# repeat:
	# take action in the lunar lander. get lots of (s, a, R(s), s')
	# store the last 10000 tuples ^
	# train the Q function on those last tuples
# assign Q = newQ
```

#### neural network architecture
it is more efficient to have the neural network output a vector that decides the action rather than passing the action in and the network deciding if it's good or not

#### epsilon greedy policy
how do you choose actions while it's still learning, the best you can do is to just pick the action that maximises Q(s,a) but it can be a good idea to 5% of the time pick an action at random. This can help you get out of local minimums.

Another way you could do this is to take a lot of random actions at the beginning and gradually drop it down.

#### Mini batch and soft updates
Mini batching:
	basically when doing linear regression with a large dataset you have to loop on a lot of examples for an average, and if there are 100 000 000 examples the it will be incredibly slow. So instead you can batch it and take the average of 1000 examples at a time.
Soft update:
	in the lunar lander example when you switch to the newQ it can be a harsh switch and can potentially be a step in the wrong direction so you can instead use some easing where you set Q = Q * 0.5 + newQ * 0.5 that way you are meeting in the middle and this can smooth out the harshness of the switch.
