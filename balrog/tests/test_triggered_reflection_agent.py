from collections import namedtuple
from types import SimpleNamespace

import balrog.agents as agents_module
from balrog.agents import AgentFactory
from balrog.agents.triggered_reflection import TriggeredReflectionAgent
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
        return LLMResponse("fake", next(self.completions), "stop", 2, 3, None)


def make_agent(*completions):
    client = FakeClient(completions)
    prompt_builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    prompt_builder.update_instruction_prompt("Task instructions")
    agent = TriggeredReflectionAgent(lambda: client, prompt_builder)
    agent.set_episode_context(80)
    return agent, client


def observation(text):
    return {
        "mission": "find the key",
        "text": {"long_term_context": text, "short_term_context": ""},
        "image": None,
    }


def actor_response(action="go north", plan="1. Explore north"):
    return f"PLAN:\n{plan}\nACTION: {action}"


def test_normal_step_uses_one_actor_call():
    agent, client = make_agent(actor_response())
    agent.set_step_context(1)

    response = agent.act(observation("a corridor"))

    assert len(client.calls) == 1
    assert response.completion == "go north"
    assert response.input_tokens == 2
    assert response.output_tokens == 3
    assert "Step 1 of 80" in client.calls[0][-1].content
    assert "Recovery reflection:\nNone." in client.calls[0][-1].content


def test_three_identical_observations_trigger_critic_then_actor():
    agent, client = make_agent(
        actor_response("go north"),
        actor_response("go north"),
        "REFLECTION: Going north is not changing the state. Try another route.",
        actor_response("go east", "1. Try the eastern route"),
    )

    agent.set_step_context(1)
    agent.act(observation("a wall ahead"))
    agent.set_step_context(2)
    agent.act(observation("a wall ahead"), prev_action="go north")
    agent.set_step_context(3)
    response = agent.act(observation("a wall ahead"), prev_action="go north")

    assert len(client.calls) == 4
    assert "Do not choose the next action" in client.calls[2][-1].content
    assert "Going north is not changing" in client.calls[3][-1].content
    assert response.completion == "go east"
    assert response.input_tokens == 4
    assert response.output_tokens == 6
    assert "TRIGGER: the same observation occurred three times" in response.reasoning
    assert "REFLECTION: Going north is not changing" in response.reasoning


def test_invalid_action_feedback_triggers_immediate_reflection():
    agent, client = make_agent(
        "REFLECTION: The command format was invalid. Return only a supported command.",
        actor_response("turn left"),
    )
    agent.set_step_context(1)

    response = agent.act(observation("Your previous output did not contain a valid action."))

    assert len(client.calls) == 2
    assert response.completion == "turn left"
    assert "explicit failure feedback" in response.reasoning


def test_textworld_not_a_verb_feedback_triggers_reflection():
    agent, client = make_agent(
        "REFLECTION: The previous text was not a command. Use one supported verb.",
        actor_response("go north"),
    )
    agent.set_step_context(1)

    response = agent.act(observation("That's not a verb I recognise."))

    assert len(client.calls) == 2
    assert response.completion == "go north"
    assert "explicit failure feedback: not a verb" in response.reasoning


def test_abab_actions_trigger_reflection():
    agent, client = make_agent(
        actor_response(),
        actor_response(),
        actor_response(),
        actor_response(),
        "REFLECTION: East-west reversal revisits the same route. Choose a third direction.",
        actor_response("go south"),
    )
    observations = ("room one", "room two", "room three", "room four", "room five")
    previous_actions = (None, "go east", "go west", "go east", "go west")

    for step, (text, previous_action) in enumerate(zip(observations, previous_actions), start=1):
        agent.set_step_context(step)
        response = agent.act(observation(text), prev_action=previous_action)

    assert len(client.calls) == 6
    assert response.completion == "go south"
    assert "A-B-A-B oscillation" in response.reasoning


def test_same_stuck_state_is_not_reflected_on_again():
    agent, client = make_agent(
        actor_response(),
        actor_response(),
        "REFLECTION: Repeating north is ineffective. Turn away from the wall.",
        actor_response("turn left"),
        actor_response("go east"),
        actor_response("go east"),
        actor_response("go east"),
        actor_response("go east"),
    )
    for step in (1, 2, 3):
        agent.set_step_context(step)
        agent.act(observation("a wall ahead"), prev_action="go north" if step > 1 else None)

    for step in (4, 5, 6, 7):
        agent.set_step_context(step)
        response = agent.act(observation("a wall ahead"), prev_action="turn left")

    assert len(client.calls) == 8
    assert response.completion == "go east"
    assert "TRIGGER:" not in response.reasoning


def test_empty_critic_response_remains_visible_in_reasoning():
    agent, client = make_agent("", actor_response("turn right"))
    agent.set_step_context(1)

    response = agent.act(observation("Your previous output did not contain a valid action."))

    assert len(client.calls) == 2
    assert response.completion == "turn right"
    assert "TRIGGER: explicit failure feedback" in response.reasoning
    assert "CRITIC RESPONSE:\n<empty>" in response.reasoning
    assert response.input_tokens == 4
    assert response.output_tokens == 6


def test_reset_clears_episode_state():
    agent, _ = make_agent(actor_response())
    agent.plan = "1. Explore"
    agent.recent_observations.append("room")
    agent.recent_actions.append("go north")
    agent.last_reflection_step = 4
    agent.last_reflection_key = ("trigger", "room")

    agent.reset()

    assert agent.plan is None
    assert list(agent.recent_observations) == []
    assert list(agent.recent_actions) == []
    assert agent.last_reflection_step is None
    assert agent.last_reflection_key is None
    assert agent.episode_step == 0
    assert agent.episode_max_steps is None


def test_parser_keeps_only_first_action_line():
    plan, action = TriggeredReflectionAgent._extract_plan_and_action(
        "PLAN:\n1. Search south\nACTION: go south\nThis should avoid the loop."
    )

    assert plan == "1. Search south"
    assert action == "go south"


def test_factory_creates_reflective_agent(monkeypatch):
    client = FakeClient([actor_response()])
    prompt_builder = HistoryPromptBuilder(max_text_history=16, max_image_history=0)
    config = SimpleNamespace(
        client=SimpleNamespace(),
        agent=SimpleNamespace(type="reflective"),
    )
    monkeypatch.setattr(agents_module, "create_llm_client", lambda _: lambda: client)
    monkeypatch.setattr(agents_module, "create_prompt_builder", lambda _: prompt_builder)

    agent = AgentFactory(config).create_agent()

    assert isinstance(agent, TriggeredReflectionAgent)
