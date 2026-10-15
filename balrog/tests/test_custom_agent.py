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


def observation(text, mission="put the blue key next to the red ball"):
    return {
        "mission": mission,
        "text": {"long_term_context": text, "short_term_context": ""},
        "image": None,
    }


def test_custom_agent_replaces_summary_and_does_not_replay_raw_history():
    agent, client = make_agent(
        "SUMMARY: first compact summary\nPLAN: 1. Find key\n2. Pick it up\nACTION: turn left",
        "SUMMARY: second compact summary\nPLAN: No changes to the plan.\nACTION: go forward",
    )

    agent.set_step_context(1)
    first = agent.act(observation("OLD_OBSERVATION_MARKER"))
    agent.set_step_context(2)
    second = agent.act(observation("NEW_OBSERVATION_MARKER"), prev_action=first.completion)

    assert first.completion == "turn left"
    assert second.completion == "go forward"
    assert agent.latest_summary == "second compact summary"
    assert agent.plan == "1. Find key\n2. Pick it up"

    second_prompt = client.calls[1][-1].content
    assert "Step 2 of 40" in second_prompt
    assert "first compact summary" in second_prompt
    assert "NEW_OBSERVATION_MARKER" in second_prompt
    assert "OLD_OBSERVATION_MARKER" not in second_prompt
    assert len(client.calls[1]) == 2


def test_custom_agent_records_an_unchanged_observation_as_failed_attempt():
    agent, client = make_agent(
        "SUMMARY: at a wall\nPLAN: Find another route\nACTION: go forward",
        "SUMMARY: still at a wall\nPLAN: Turn away\nACTION: turn left",
    )

    agent.set_step_context(1)
    first = agent.act(observation("a wall 1 step forward"))
    agent.set_step_context(2)
    agent.act(observation("a wall 1 step forward"), prev_action=first.completion)

    assert list(agent.failed_attempts) == ["'go forward' left the visible observation unchanged"]
    assert "'go forward' left the visible observation unchanged" in client.calls[1][-1].content


def test_reset_clears_all_episode_memory():
    agent, _ = make_agent("SUMMARY: state\nPLAN: a plan\nACTION: look")
    agent.set_step_context(3)
    agent.act(observation("room"))
    agent.failed_attempts.append("failed route")

    agent.reset()

    assert agent.plan is None
    assert agent.latest_summary is None
    assert agent.objective is None
    assert list(agent.failed_attempts) == []
    assert agent.episode_step == 0
    assert agent.episode_max_steps is None


def test_extract_sections_accepts_labelled_and_tagged_actions():
    summary, plan, action = CustomAgent._extract_sections(
        "SUMMARY: current state\nPLAN: explore north\nACTION: go north\nextra explanation"
    )
    assert (summary, plan, action) == ("current state", "explore north", "go north")

    summary, plan, action = CustomAgent._extract_sections("reasoning\n<|ACTION|>take coin<|END|>")
    assert summary is None
    assert plan is None
    assert action == "take coin"
