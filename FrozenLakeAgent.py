from ReplayMemory import ReplayMemory, Transition
from pennylane import numpy as np
import gymnasium as gym
import torch
from HQNN import HQNN
import matplotlib.pyplot as plt
from PenaltyWrapper import PenaltyWrapper


class FrozenLakeAgent:
    def __init__(self, batch_size, gamma, epsilon, tau, lr, memory_size):
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.lr = lr

        self.n_states = 16
        self.n_actions = 4

        self.target_circuit = HQNN()
        self.policy_circuit = HQNN()

        self.target_circuit.load_state_dict(self.policy_circuit.state_dict())

        # self.optimizer = torch.optim.Adam(self.policy_circuit.parameters(), lr=lr)
        self.optimizer = torch.optim.RMSprop(self.policy_circuit.parameters(), lr=lr, alpha=0.99, eps=1e-8,
                                             weight_decay=0, momentum=0, centered=False)

        self.memory = ReplayMemory(memory_size)
        self.max_steps = 2500

    def select_action(self, state):
        sample = np.random.random()

        epsilon_threshold = (self.epsilon / self.n_actions) + 1 - self.epsilon
        #if epsilon_threshold > 0.97:
        #    epsilon_threshold = 0.97

        if sample < epsilon_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_circuit(state))
        else:
            return torch.tensor(np.random.randint(0, self.n_actions))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough experience for training

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_circuit(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_circuit(non_final_next_states).max(1).values.float()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_circuit.parameters(), 100)
        self.optimizer.step()

    def train(self, n_episodes, render=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)
        #env = PenaltyWrapper(env)
        rewards_per_episode = np.zeros(n_episodes)
        for episode in range(n_episodes):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for step in range(self.max_steps):
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())

                reward = torch.tensor([reward], dtype=torch.float32)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.learn()

                if done:
                    rewards_per_episode[episode] = reward.item()
                    if rewards_per_episode[episode] > 0:
                        print(f"Episode {episode} finished after {step + 1} steps with reward {reward.item()}")
                        self.epsilon = self.epsilon / (1 + (episode / 100))
                        threshold = (self.epsilon / self.n_actions) + 1 - self.epsilon
                        print(f"\t {100 - 100*threshold:.3f}% chance of selecting a random action")
                    # print(f"Episode {episode} finished after {step} steps with reward {reward.item()}")
                    break

            if episode % 20 == 0:
                target_circuit_state_dict = self.target_circuit.state_dict()
                policy_circuit_state_dict = self.policy_circuit.state_dict()

                #self.target_circuit.load_state_dict(policy_circuit_state_dict)
                for key in policy_circuit_state_dict:
                    target_circuit_state_dict[key] = (policy_circuit_state_dict[key] * self.tau
                                                      + target_circuit_state_dict[key] * (1 - self.tau))
        env.close()

        plt.plot(rewards_per_episode)
        plt.savefig('out.png')

    def test(self):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='human')
        env = PenaltyWrapper(env)
        for episode in range(1):
            state = env.reset()[0]
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken
            # 200 actions (truncated).
            while not terminated and not truncated:
                # Select best action
                with torch.no_grad():
                    action = torch.argmax(self.policy_circuit([state])).item()

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)
                print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        env.close()
