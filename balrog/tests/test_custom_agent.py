from collections import namedtuple

from balrog.agents.custom import CustomAgent
from balrog.prompt_builder.history import HistoryPromptBuilder


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


class FakeClient:
    def __init__(self, completions):
        self.completions = iter(completions)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return LLMResponse("fake", next(self.completions), "stop", 1, 1, None)


def make_agent(*completions):
    client = FakeClient(completions)
    prompt_builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    prompt_builder.update_instruction_prompt("Task instructions")
    agent = CustomAgent(lambda: client, prompt_builder)
    agent.set_episode_context(40)
    return agent, client


def observation(text):
    return {
        "mission": "put the blue key next to the red ball",
        "text": {"long_term_context": text, "short_term_context": ""},
        "image": None,
    }


def test_custom_agent_reuses_history_and_replaces_the_complete_plan():
    agent, client = make_agent(
        "PLAN:\n1. Find the key\n2. Pick it up\nACTION: turn left",
        "PLAN:\n1. Pick up the key\n2. Place it by the ball\nACTION: go forward",
    )

    agent.set_step_context(1)
    first = agent.act(observation("OLD_OBSERVATION_MARKER"))
    agent.set_step_context(2)
    second = agent.act(observation("NEW_OBSERVATION_MARKER"), prev_action=first.completion)

    assert first.completion == "turn left"
    assert second.completion == "go forward"
    assert second.reasoning == "1. Pick up the key\n2. Place it by the ball"
    assert agent.plan == second.reasoning

    second_prompt = "\n".join(message.content for message in client.calls[1])
    assert "Step 2 of 40" in second_prompt
    assert "1. Find the key\n2. Pick it up" in second_prompt
    assert "OLD_OBSERVATION_MARKER" in second_prompt
    assert "NEW_OBSERVATION_MARKER" in second_prompt
    assert "at most three numbered subgoals" in second_prompt
    assert "under 80 words" in second_prompt


def test_reset_clears_plan_history_and_step_context():
    agent, _ = make_agent("PLAN:\n1. Explore north\nACTION: go north")
    agent.set_step_context(3)
    agent.act(observation("a room"))

    agent.reset()

    assert agent.plan is None
    assert agent.episode_step == 0
    assert agent.episode_max_steps is None
    assert [message.content for message in agent.prompt_builder.get_prompt()] == ["Task instructions"]


def test_parser_keeps_only_the_first_action_line():
    plan, action = CustomAgent._extract_plan_and_action(
        "PLAN:\n1. Explore north\n2. Take the coin\nACTION: go north\nThis exit is unexplored."
    )

    assert plan == "1. Explore north\n2. Take the coin"
    assert action == "go north"


def test_parser_fails_safe_when_labels_are_missing():
    plan, action = CustomAgent._extract_plan_and_action("I think going north is best.")

    assert plan is None
    assert action == ""
