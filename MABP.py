import numpy as np
import gym
from gym import spaces

class MultiAgentBanditEnv(gym.Env):
    def __init__(self, num_agents=2, num_actions=2, reward_matrix=None):
        super(MultiAgentBanditEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Action space: each agent chooses an action from 0 to num_actions-1
        self.action_space = spaces.MultiDiscrete([num_actions] * num_agents)
        
        # Observation space: observe the action of the other agent(s)
        self.observation_space = spaces.MultiDiscrete([num_actions] * (num_agents - 1))
        
        # Define the reward matrix for the prisoner's dilemma
        if reward_matrix is None:
            self.reward_matrix = np.array([
                [[3, 0], [5, 1]],
                [[0, 5], [1, 3]]
            ])
        else:
            self.reward_matrix = reward_matrix
        
        self.reset()
    
    def reset(self):
        self.state = None
        return self._get_observation()
    
    def step(self, actions):
        assert len(actions) == self.num_agents, "Number of actions must match number of agents."
        
        # Calculate rewards for each agent
        rewards = np.zeros(self.num_agents)
        
        # Use appropriate indexing to get scalar values
        reward_value = self.reward_matrix[actions[0], actions[1]]
        for i in range(self.num_agents):
            rewards[i] = reward_value
        
        self.state = actions
        
        return self._get_observation(), rewards, False, {}
    
    def _get_observation(self):
        if self.state is None:
            return [0] * (self.num_agents - 1)
        else:
            return [self.state[i] for i in range(1, self.num_agents)]
        
    def render(self, mode='human'):
        if self.state is None:
            print("Environment not initialized.")
        else:
            print(f"Agent actions: {self.state}")

# Example usage
if __name__ == "__main__":
    env = MultiAgentBanditEnv()
    obs = env.reset()
    done = False

    while not done:
        actions = env.action_space.sample()
        obs, rewards, done, info = env.step(actions)
        env.render()
        print(f"Observation: {obs}, Rewards: {rewards}")
