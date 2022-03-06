# Double_DQN_CartPole
Double Deep Q Network (DQN) approach to solving the OpenAI CartPole problem with a Fully-Connected Neural Network (FCNN).

# Cart-Pole Problem
The cart-pole problem is a classic highly non-linear control problem, which involves moving a cart left or right to simultaneously balance an attached pole in the upright position.  The four states in consideration include the cart position, cart velocity, pole angle and pole angular velocity.  The action space includes imparting a force on the cart to move it either left or right depending on the current state.  The criteria for balancing the cart involves (1) ensuring the pole angle doesn’t exceed 12 degrees from the center in either direction and (2) ensuring the cart doesn’t move 2.4 units from the center in either direction.  A reward of +1 is given for every time step the cart meets the criteria listed above.  The episode will terminate when either (1) or (2) are violated and if the episode exceeds 200 time steps.  The criteria for solving the problem, as defined by OpenAI, is described as obtaining an average reward of greater than or equal to 195 over 100 consecutive episodes.

# Training Results
![dqn_agent_train_2linlayers_cv0](https://user-images.githubusercontent.com/101025359/156903673-ae9df6de-fff2-4f3d-8d10-e6c0466182ae.png)

# Testing Results
![dqn_agent_test_avg_2linlayers_cv0](https://user-images.githubusercontent.com/101025359/156903678-a007f111-c144-4061-a86a-1d6e2325f28c.png)

![dqn_agent_test_render_2linlayers_cv0](https://user-images.githubusercontent.com/101025359/156903670-7bf00108-2360-4f28-987c-7ee9ab511620.gif)

# Overall Results
From the training results, it can be seen that the Double DQN agent reaches the goal of 195 more frequently towards the latter half of the 200 training episodes.  The histogram plots illustrates the agent achieving over 190 rewards over 12 times out of the 200 training episodes.  The training results also show the upward trend in rewards as the number of episodes increase, which indicates that the agent is learning along the way.  After training, the agent was tested on the OpenAI Cart-Pole v0 environment with a goal of achieving an average of at least 195 rewards over 100 testing episodes.  The training results show that the trained agent is able to surpass the goal in all cases.  In fact, the agent achieves an average of over 199 rewards over the span of 100 episodes.  The testing results show that a Double DQN agent with a simple FCNN architecture can successfully solve the cart-pole problem given the criteria defined above.
