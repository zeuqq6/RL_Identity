import numpy as np
import gym
from gym import spaces

class MultiArmedBanditEnv(gym.Env):
    def __init__(self, num_agents=10, history_length = 5):
        super(MultiArmedBanditEnv, self).__init__()
        self.num_agents = num_agents
        self.history_length = history_length
        self.action_space = spaces.Discrete(2)  # 0: Not Cooperate, 1: Cooperate
        self.observation_space = spaces.Box(low=0, high=1, shape=(history_length * num_agents,), dtype=np.float32)
        
        self.state = np.zeros(self.num_agents)
        self.history = np.zeros((history_length, num_agents))
        self.reset()

    def reset(self):
        self.state = np.random.rand(self.num_agents)
         # initialize history with the current state repeated
        self.history = np.tile(self.state, (self.history_length, 1))
        return self.history.flatten()
        # return self.state

    def step(self, actions):
        rewards = np.zeros(self.num_agents)
        new_state = np.zeros(self.num_agents)
        
        # iterate through pairs of agents to evaluate their actions
        for i in range(0, self.num_agents, 2):
            action1 = actions[i]
            action2 = actions[i+1]

            if action1 == 1 and action2 == 1: #if  both cooperate
                rewards[i] = rewards[i+1] = 5
            elif action1 == 0 and action2 == 0: # if neither cooperate
                rewards[i] = rewards[i+1] = 1
            elif action1 == 1 and action2 == 0:
                rewards[i] = 0
                rewards[i+1] = 10
            else:
                rewards[i] = 10
                rewards[i+1] = 0
            
            new_state[i] = action1
            new_state[i+1] = action2

        # self.state = np.random.rand(self.num_agents)
        # update history
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :] = actions
        done = True  # One-step game
        return self.history.flatten(), rewards, done, {}

    def render(self, mode='human', close=False):
        pass

class Agent:
    def __init__(self, color, cooperation_probability):
        self.color = color
        self.cooperation_probability = cooperation_probability
        self.rewards = []

    def decide(self):
        return np.random.rand() < self.cooperation_probability

class QLearningAgent:
    def __init__(self, num_agents, red_coop_prob, blue_coop_prob, history_length=10, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, min_epsilon=0.1):
        self.num_agents = num_agents
        self.history_length = history_length
        self.agents = self._initialize_agents(red_coop_prob, blue_coop_prob)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon

                # Limit history length to 10
        if self.history_length > 10:
            raise ValueError("History length must be 10 or less to manage Q-table size.")

        # Calculate number of states and actions with limited history
        self.num_states = 2 ** (self.num_agents * self.history_length)
        self.num_actions = 2 ** self.num_agents
        
        print(f"num_states: {self.num_states}, num_actions: {self.num_actions}")
        
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def _initialize_agents(self, red_coop_prob, blue_coop_prob):
        agents = []
        for i in range(self.num_agents):
            if i < self.num_agents // 2:
                agents.append(Agent('red', red_coop_prob))
            else:
                agents.append(Agent('blue', blue_coop_prob))
        return agents

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2, size=self.num_agents)
        else:
            state_idx = self._state_to_index(state)
            action_idx = np.argmax(self.q_table[state_idx])
            return self._index_to_actions(action_idx)

    def update_q_table(self, state, actions, reward, next_state):
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        action_idx = self._actions_to_index(actions)
        
        # normalize reward to improve stability
        normalized_reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        
        q_target = normalized_reward + self.gamma * np.max(self.q_table[next_state_idx])
        self.q_table[state_idx, action_idx] += self.lr * (q_target - self.q_table[state_idx, action_idx])
        # q_predict = self.q_table[state_idx, action_idx]
        # q_target = reward + self.gamma * np.max(self.q_table[next_state_idx])
        # self.q_table[state_idx, action_idx] += self.lr * (q_target)

    def _state_to_index(self, state):
        return int("".join(map(str, (state >= 0.5).astype(int))), 2)

    def _actions_to_index(self, actions):
        return int("".join(map(str, actions)), 2)

    def _index_to_actions(self, index):
        return [int(x) for x in np.binary_repr(index, width=self.num_agents)]

    # reduces the exploration rate over time; shift from exploration to exploitation
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # self.epsilon *= self.epsilon_decay