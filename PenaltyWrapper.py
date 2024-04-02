import gymnasium as gym


class PenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.penalty = -0.01
        # Assuming FrozenLake uses 'G' to represent the goal state
        self.goal_states = {state for state, letter in enumerate(env.unwrapped.desc.flatten()) if letter == b'G'}
        self.current_penalty = 0  # Track accumulated penalty

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        if state not in self.goal_states:
            self.current_penalty += self.penalty  # Increase penalty
            reward += self.current_penalty  # Apply accumulated penalty

        return state, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.current_penalty = 0  # Reset penalty at the start of an episode
        return self.env.reset(**kwargs)
