"""Executable scenarios for the Reflective + Robust CoT hybrid agent.

These tests focus on the contract between the two mechanisms: Robust CoT
chooses every action, while the critic is called only after observable failure.
"""

from collections import namedtuple
from types import SimpleNamespace

import balrog.agents as agents_module
from balrog.agents import AgentFactory
from balrog.agents.reflective_robust_cot import ReflectiveRobustCoTAgent
from balrog.prompt_builder.history import HistoryPromptBuilder


LLMResponse = namedtuple(
    "LLMResponse",
    ["model_id", "completion", "stop_reason", "input_tokens", "output_tokens", "reasoning"],
)


class FakeClient:
    """Return scripted responses while retaining each prompt for inspection."""

    def __init__(self, completions):
        self.completions = iter(completions)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return LLMResponse("fake", next(self.completions), "stop", 2, 3, None)


def make_agent(*completions):
    client = FakeClient(completions)
    prompt_builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    prompt_builder.update_instruction_prompt("Task instructions")
    agent = ReflectiveRobustCoTAgent(lambda: client, prompt_builder)
    agent.set_episode_context(80)
    return agent, client


def observation(text):
    return {
        "mission": "find the key",
        "text": {"long_term_context": text, "short_term_context": ""},
        "image": None,
    }


def robust_response(action="go north", plan="1. Explore north", prefix=""):
    return f"{prefix}PLAN:\n{plan}\n<|ACTION|>{action}<|END|>"


def test_normal_step_uses_robust_actor_without_critic():
    agent, client = make_agent(robust_response(prefix="The northern exit is unexplored.\n"))
    agent.set_step_context(1)

    response = agent.act(observation("a corridor"))

    assert len(client.calls) == 1
    assert response.completion == "go north"
    assert agent.plan == "1. Explore north"
    assert "Step 1 of 80" in client.calls[0][-1].content
    assert "predict its immediate effect" in client.calls[0][-1].content
    assert "<|ACTION|>" in client.calls[0][-1].content


def test_strict_parser_never_sends_reasoning_to_environment():
    agent, _ = make_agent(robust_response("turn left", prefix="I should avoid the wall.\n"))
    agent.set_step_context(1)

    response = agent.act(observation("a wall ahead"))

    assert response.completion == "turn left"
    assert "I should avoid the wall" in response.reasoning


def test_missing_action_delimiters_fail_safe():
    agent, _ = make_agent("PLAN:\n1. Explore north\nACTION: go north")
    agent.set_step_context(1)

    response = agent.act(observation("a corridor"))

    assert response.completion == ""
    assert agent.plan == "1. Explore north"


def test_repeated_observation_runs_critic_then_robust_actor():
    agent, client = make_agent(
        robust_response("go north"),
        robust_response("go north"),
        "REFLECTION: Going north leaves the state unchanged. Choose another route.",
        robust_response("turn right", "1. Leave the blocked route"),
    )

    agent.set_step_context(1)
    agent.act(observation("a wall ahead"))
    agent.set_step_context(2)
    agent.act(observation("a wall ahead"), prev_action="go north")
    agent.set_step_context(3)
    response = agent.act(observation("a wall ahead"), prev_action="go north")

    assert len(client.calls) == 4
    assert response.completion == "turn right"
    assert response.input_tokens == 4
    assert response.output_tokens == 6
    assert "Going north leaves the state unchanged" in client.calls[3][-1].content
    assert "TRIGGER: the same observation occurred three times" in response.reasoning


def test_textworld_rejection_is_remembered_before_robust_action():
    agent, client = make_agent(
        "REFLECTION: Open the passageway before trying to move through it.",
        robust_response("open passageway", "1. Open the blocked passage"),
    )
    agent.last_observation_signature = "cookery"
    agent.set_step_context(2)

    response = agent.act(
        observation("You have to open the passageway first."),
        prev_action="go north",
    )

    assert response.completion == "open passageway"
    assert "go north -> You have to open the passageway first." in client.calls[0][-1].content
    assert "Open the passageway before trying" in client.calls[1][-1].content


def test_productive_repetition_does_not_call_critic():
    agent, client = make_agent(
        robust_response("go forward"),
        robust_response("go forward"),
        robust_response("go forward"),
    )

    for step, distance in enumerate((3, 2, 1), start=1):
        agent.set_step_context(step)
        response = agent.act(
            observation(f"target {distance} steps ahead"),
            prev_action="go forward" if step > 1 else None,
        )

    assert len(client.calls) == 3
    assert response.completion == "go forward"
    assert "TRIGGER:" not in response.reasoning


def test_factory_creates_hybrid_agent(monkeypatch):
    client = FakeClient([robust_response()])
    prompt_builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    config = SimpleNamespace(
        client=SimpleNamespace(),
        agent=SimpleNamespace(type="reflective_robust_cot"),
    )
    monkeypatch.setattr(agents_module, "create_llm_client", lambda _: lambda: client)
    monkeypatch.setattr(agents_module, "create_prompt_builder", lambda _: prompt_builder)

    agent = AgentFactory(config).create_agent()

    assert isinstance(agent, ReflectiveRobustCoTAgent)
