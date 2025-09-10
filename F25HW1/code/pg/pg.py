#! python3

import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass

    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        # END STUDENT SOLUTION
        pass

    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        num_steps = len(rewards)

        if self.mode == "REINFORCE":
            
            # compute discounted rewards at each time step
            G_t = []
            for t in range(num_steps):
                g = sum([self.gamma ** (k - t) * rewards[k] for k in range(t, num_steps)]) # compute G_t which is just a scalar the incorporates all the future discounted rewards starting from time step t until the end
                G_t.append(g)
            G_t = torch.tensor(G_t, dtype=torch.float32).to(self.device) # convert G_t to a tensor

            # compute the probability of the actions taken given the states
            policy_probs = self.forward(states_tensor)[0] # get the policy probabilities for all time steps
            probs_taken = policy_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) # get the probabilities of the actions taken by using actions_tensor to index into policy_probs
            log_probs = torch.log(probs_taken) # take the log of the policy probabilities

            # compute the loss function and update the actor network 
            loss = -torch.mean(log_probs * G_t) # compute the loss function
            self.optimizer_actor.zero_grad() # zero the gradients
            loss.backward() # backpropagate the loss
            self.optimizer_actor.step() # update the actor network

        elif self.mode == "REINFORCE_WITH_BASELINE":
            pass
        elif self.mode == "A2C":
            pass 
        
        # END STUDENT SOLUTION
        pass

    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for ep in range(num_episodes):
            states, actions, rewards = [], [], []

            state, _ = env.reset()
            for step in range(max_steps):
                action = self.get_action(state, stochastic=train)
                next_state, reward, terminated, truncated, _ = env.step(action) 

                # store state, action, reward
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                if terminated or truncated:
                    break

                # transition to next state
                state = next_state
            
            # store total reward for this episode
            total_rewards.append(sum(rewards))

            if train:
                self.train(states, actions, rewards)

        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name,
    agents,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    if agents[0].n != 0:
        graph_name += "_" + str(agents[0].n)

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(f"./graphs/{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode
    n = args.n
    agent = PolicyGradient(
        state_size=state_size,
        action_size=action_size,
        mode=mode,
        n=n,
        device=device,
    )

    # TODO: come back and finish this
    agent.run(
        env=env,
        max_steps=args.max_steps,
        num_episodes=args.num_episodes,
        train=True,
    )
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
