class BaseAgent:
    """Base class for agents using prompt-based interactions."""

    def __init__(self, client_factory, prompt_builder):
        """Initialize the agent with a client and prompt builder."""
        self.client = client_factory()
        self.prompt_builder = prompt_builder
        self.episode_step = 0
        self.episode_max_steps = None

    def act(self, obs):
        """Generate an action based on the observation."""
        raise NotImplementedError

    def update_prompt(self, observation, action):
        """Update the prompt with the observation and action."""
        self.prompt_builder.update_observation(observation)
        self.prompt_builder.update_action(action)

    def reset(self):
        """Reset the prompt builder."""
        self.prompt_builder.reset()
        self.episode_step = 0
        self.episode_max_steps = None

    def set_episode_context(self, max_steps):
        """Provide the step budget for the current episode."""
        self.episode_max_steps = max_steps

    def set_step_context(self, step):
        """Provide the one-based index of the next action decision."""
        self.episode_step = step
