#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
os.makedirs('./graphs', exist_ok=True)


class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.queue = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION  

    def __len__(self):
        return len(self.queue)
    
    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        batch = [random.choice(self.queue) for _ in range(self.batch_size)] # sample with replacement
        # END STUDENT SOLUTION
        return batch


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.queue.append(transition)
        # END STUDENT SOLUTION


class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, double_dqn, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = double_dqn

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, self.action_size), # Q values can be negative, use just linear layer
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.q_net = q_net_init()
        self.target_net = q_net_init()

        self.q_net.to(device)
        self.target_net.to(device)

        self.buffer = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        self.optimizer_q = optim.Adam(self.q_net.parameters(), lr=lr_q_net)
       
        
        # END STUDENT SOLUTION


    def forward(self, state, new_state):
        # calculate q value and target
        # use the correct network for the target based on self.double_dqn
        # BEGIN STUDENT SOLUTION
        # NOTE: This was wrong for me before. 
        # You should always have a seperate net /delayed for q_values and target. Having one "Frozen" version is important to ensure stability. Otherwise you need many more samples
        q_values = self.q_net(state)

        # Target Q-values for next states (no gradient needed)
        with torch.no_grad():
            target_q_values = self.target_net(new_state)

        return q_values, target_q_values
        # END STUDENT SOLUTION
       

    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        
        # Convert state to tensor with batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
    
        # Get Q-values for the state
        q_values, _ = self.forward(state_tensor, state_tensor)  # new_state ignored for single step

        # Epsilon-greedy: random action with probability epsilon
        if stochastic and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)

        # Otherwise, choose the action with highest Q-value
        action = torch.argmax(q_values, dim=-1).item()
        return action
    
        # END STUDENT SOLUTION
        

    def train(self, batch):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION

        # Unpack the batch
        states, actions, rewards, new_states, terminated = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        new_states_tensor = torch.tensor(new_states, dtype=torch.float32).to(self.device)
        terminated_tensor = torch.tensor(terminated, dtype=torch.float32).to(self.device)

        # Forward pass (get q values and target values)
        q_values, target_q_values = self.forward(states_tensor, new_states_tensor)

        # Compute targets (Double DQN vs DQN)
        with torch.no_grad():
            if self.double_dqn:
                # Use online network to pick the best next action (You should take the q values from our online network at the next state and then do arg max on that so our online network is responsible for action selection)
                # Online network selects the best action for next states
                online_next_q = self.q_net(new_states_tensor)
                best_actions = torch.argmax(online_next_q, dim=-1) 
                next_q = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1) # target network evaluates this action
            else:
                # Regular DQN: take max Q-value from target network
                next_q = torch.max(target_q_values, dim=-1)[0]

            yi = rewards_tensor + self.gamma * next_q * (1 - terminated_tensor)

        # Select q-values corresponding to the actions we actually took
        relevant_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute MSE loss
        loss = F.mse_loss(relevant_q_values, yi)

        # Backprop
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()
        # END STUDENT SOLUTION

    def do_burn_in(self, env):
        # Collect multiple transitions --> One long episode
        # Init Replay Buffer 
        while len(self.buffer) < self.burn_in:
           
            state, _ = env.reset()

            # Use Uniform random policy so its not bias to our current policy. NOTE: I was sampling from the policy using e-greedy. But this biases our burn in to be policy actions. Not good.
            action =np.random.randint(0, self.action_size)
            next_state, reward, terminated, truncated, _ = env.step(action) 

            # Store State action Reward
            self.buffer.append([state, action, reward, next_state, terminated or truncated])

            # transition to next state
            state = next_state
        
            if terminated or truncated:
                state, _ = env.reset()
        return

    def test_curr_policy(self, env, num_episodes, max_steps):
        # Run the "frozen" policy for e episodes and evaluate the reward
        reward_per_episode = []
        for ep in range(num_episodes):
            state, _ = env.reset()
            total_ep_reward = 0 
            for t in range(max_steps):
                action = self.get_action(state, stochastic=False)
                next_state, reward, terminated, truncated, _ = env.step(action) 
                state = next_state
                total_ep_reward += reward
                
                if terminated or truncated:
                    break 
            
            reward_per_episode.append(total_ep_reward)
        
        mean_undiscounted_return = np.mean(np.array(reward_per_episode))
        return mean_undiscounted_return
            


    def run(self, env, max_steps, num_episodes, test_every):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        
        # Collect trials
        self.do_burn_in(env)

        # Collect a bunch of experience
        for ep in range(num_episodes):
            
            # Establish if we should test this episode 
            test_trial = ep % test_every == 0

            # Test if necessary
            if test_trial:
                eval_episodes = 20 # Per hw
                mean_undiscounted_return = self.test_curr_policy(env, eval_episodes, max_steps)
                print(f"Ep {ep} | Trial {ep // test_every} | Reward {mean_undiscounted_return}")
                total_rewards.append(mean_undiscounted_return)

            # Update target network periodically (every x episodes)
            if ep % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            # Collect more trainsitions/steps and train our policy
            state, _ = env.reset()
            for step in range(max_steps):

                # Take action from policy or eps-greedy 
                action = self.get_action(state, stochastic=True)
                next_state, reward, terminated, truncated, _ = env.step(action) 

                # Store State action Reward
                self.buffer.append([state, action, reward, next_state, terminated or truncated])

                # transition to next state
                state = next_state
                
                if terminated or truncated:
                    break

                # train, burn in should be reached already
                random_batch = self.buffer.sample_batch()
                self.train(random_batch)

        # END STUDENT SOLUTION
        return total_rewards 
    


def graph_agents(
    graph_name, mean_undiscounted_returns, test_frequency, max_steps, num_episodes
):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    
    D = np.array(mean_undiscounted_returns)  # shape: [num_runs, num_checkpoints]
    print(D.shape)
    # Compute stats across runs
    average_total_rewards = D.mean(axis=0)
    min_total_rewards = D.min(axis=0)
    max_total_rewards = D.max(axis=0)

    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * test_frequency for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument(
        "--test_frequency",
        type=int,
        default=100,
        help="Number of training episodes between test episodes",
    )
    parser.add_argument("--double_dqn", action="store_true", help="Use Double DQN")
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION

    env_name = args.env_name
    num_runs = args.num_runs
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    test_frequency = args.test_frequency
    double_dqn = args.double_dqn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    D = [] # matrix of results

    for run_idx in range(num_runs):
        print(f"\n=== Starting run {run_idx+1}/{num_runs} ===")
        env = gym.make(env_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Initialize agent
        agent = DeepQNetwork(
            state_size=state_size,
            action_size=action_size,
            double_dqn=double_dqn,
            device=device
        )

        # Train agent
        rewards = agent.run(env, max_steps=max_steps,
                             num_episodes=num_episodes, 
                             test_every=test_frequency)
        D.append(rewards)

        env.close()

    # Graph results using D
    name = f"{env_name}_DDQN" if double_dqn else f"{env_name}_DQN"
    graph_agents(
        graph_name=name,
        mean_undiscounted_returns=D,  
        test_frequency=test_frequency,
        max_steps=max_steps,
        num_episodes=num_episodes
    )

    # Graph the results
    # END STUDENT SOLUTION



if __name__ == '__main__':
    main()

