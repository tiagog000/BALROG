from collections import namedtuple

from omegaconf import OmegaConf

from balrog.agents.base import BaseAgent
from balrog.evaluator import Evaluator


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


class FakePromptBuilder:
    def __init__(self):
        self.instruction = None

    def reset(self):
        self.instruction = None

    def update_instruction_prompt(self, instruction):
        self.instruction = instruction


class TrackingAgent(BaseAgent):
    def __init__(self):
        super().__init__(lambda: None, FakePromptBuilder())
        self.contexts = []

    def act(self, obs, prev_action=None):
        self.contexts.append((self.episode_step, self.episode_max_steps))
        return LLMResponse("fake", "go forward", "stop", 1, 1, None)


class FakeEnv:
    max_steps = 3
    failed_candidates = []

    def __init__(self):
        self.steps = 0

    @staticmethod
    def _observation():
        return {
            "mission": "reach the target",
            "text": {"long_term_context": "a wall 2 steps left", "short_term_context": ""},
            "image": None,
        }

    def reset(self, seed=None):
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        return self._observation(), 0.0, self.steps == 2, False, {}

    def get_instruction_prompt(self, instructions=None):
        return f"Complete: {instructions}"

    @staticmethod
    def check_action_validity(action):
        return action

    @staticmethod
    def get_stats():
        return {"progression": 0.0}


def test_evaluator_provides_real_step_context(monkeypatch, tmp_path):
    env = FakeEnv()
    monkeypatch.setattr("balrog.evaluator.make_env", lambda *args, **kwargs: env)

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.env_name = "babyai"
    evaluator.output_dir = str(tmp_path)
    evaluator.max_steps_per_episode = None
    evaluator.config = OmegaConf.create(
        {
            "envs": {"env_kwargs": {"seed": 1001}},
            "eval": {"feedback_on_invalid_action": False, "save_images": False},
            "agent": {"type": "tracking"},
            "client": {"client_name": "fake"},
        }
    )

    agent = TrackingAgent()
    result = evaluator.run_episode("context-test", agent, episode_idx=0)

    assert agent.contexts == [(1, 3), (2, 3)]
    assert result["num_steps"] == 2
    assert (tmp_path / "babyai" / "context-test" / "context-test_run_00.json").exists()
