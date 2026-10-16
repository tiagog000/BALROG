import re
from collections import deque

from balrog.agents.base import BaseAgent


class TriggeredReflectionAgent(BaseAgent):
    """Plan normally and request one critique only after observable failure."""

    REFLECTION_COOLDOWN_STEPS = 3
    FAILURE_MARKERS = (
        "previous output did not contain a valid action",
        "not a verb",
    )

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.plan = None
        self.recent_observations = deque(maxlen=3)
        self.recent_actions = deque(maxlen=4)
        self.last_reflection_step = None
        self.last_reflection_key = None

    def reset(self):
        """Clear all episode-specific planning and reflection state."""
        super().reset()
        self.plan = None
        self.recent_observations.clear()
        self.recent_actions.clear()
        self.last_reflection_step = None
        self.last_reflection_key = None

    def act(self, obs, prev_action=None):
        """Reflect when triggered, then update the plan and choose one action."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
            self.recent_actions.append(self._normalise(prev_action))
        self.prompt_builder.update_observation(obs)

        observation = self._observation_text(obs)
        observation_signature = self._normalise(observation)
        if self.recent_observations and observation_signature != self.recent_observations[-1]:
            self.last_reflection_key = None
        self.recent_observations.append(observation_signature)
        trigger = self._reflection_trigger(observation)
        reflection_key = (trigger, observation_signature)

        critic_response = None
        reflection = None
        if trigger and reflection_key != self.last_reflection_key and self._cooldown_complete():
            critic_response = self.client.generate(self._critic_prompt(trigger))
            reflection = self._extract_reflection(critic_response.completion)
            self.last_reflection_step = self.episode_step
            self.last_reflection_key = reflection_key

        actor_response = self.client.generate(self._actor_prompt(reflection))
        plan, action = self._extract_plan_and_action(actor_response.completion)
        if plan:
            self.plan = plan

        reasoning_parts = []
        if critic_response is not None:
            raw_critique = critic_response.completion or "<empty>"
            reasoning_parts.extend((f"TRIGGER: {trigger}", f"CRITIC RESPONSE:\n{raw_critique}"))
            if reflection:
                reasoning_parts.append(f"REFLECTION: {reflection}")
        reasoning_parts.append(f"PLAN RESPONSE:\n{actor_response.completion}")

        input_tokens = actor_response.input_tokens
        output_tokens = actor_response.output_tokens
        if critic_response is not None:
            input_tokens += critic_response.input_tokens
            output_tokens += critic_response.output_tokens

        return actor_response._replace(
            completion=action,
            reasoning="\n\n".join(reasoning_parts),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _critic_prompt(self, trigger):
        messages = self.prompt_builder.get_prompt()
        recent_actions = ", ".join(self.recent_actions) or "None"
        plan = self.plan or "No plan yet."
        instructions = f"""
You are a critic evaluating an agent that appears to be stuck.

Reflection trigger:
{trigger}

Current plan:
{plan}

Recent actions:
{recent_actions}

Identify one mistaken assumption or ineffective strategy and one specific change to make.
Base the diagnosis only on the available evidence. Do not choose the next action.
Keep the response under 30 words.

Return exactly:
REFLECTION: <one concise lesson>
        """.strip()
        messages[-1].content += "\n\n" + instructions
        return messages

    def _actor_prompt(self, reflection):
        messages = self.prompt_builder.get_prompt()
        step = self.episode_step or "?"
        max_steps = self.episode_max_steps if self.episode_max_steps is not None else "?"
        plan = self.plan or "No plan yet."
        lesson = reflection or "None."
        instructions = f"""
Step {step} of {max_steps}.

Current plan:
{plan}

Recovery reflection:
{lesson}

Update the plan from the current observation, recent action history, and any recovery reflection.
The complete plan must contain at most three numbered subgoals. Keep the response under 80 words.

Return exactly:
PLAN:
1. <subgoal>
ACTION: <one complete valid game command, including every required argument>
        """.strip()
        messages[-1].content += "\n\n" + instructions
        return messages

    def _reflection_trigger(self, observation):
        lower_observation = observation.lower()
        for marker in self.FAILURE_MARKERS:
            if marker in lower_observation:
                return f"explicit failure feedback: {marker}"

        if len(self.recent_observations) == 3 and len(set(self.recent_observations)) == 1:
            return "the same observation occurred three times"

        if len(self.recent_actions) == 4:
            first, second, third, fourth = self.recent_actions
            if first == third and second == fourth and first != second:
                return "the recent actions form an A-B-A-B oscillation"

        return None

    def _cooldown_complete(self):
        if self.last_reflection_step is None:
            return True
        return self.episode_step - self.last_reflection_step > self.REFLECTION_COOLDOWN_STEPS

    @staticmethod
    def _observation_text(obs):
        if not isinstance(obs, dict):
            return str(obs)
        text = obs.get("text", {})
        if not isinstance(text, dict):
            return str(text)
        parts = (text.get("short_term_context", ""), text.get("long_term_context", ""))
        return "\n".join(str(part).strip() for part in parts if part and str(part).strip())

    @staticmethod
    def _normalise(value):
        return re.sub(r"\s+", " ", str(value)).strip().lower()

    @staticmethod
    def _extract_reflection(response_text):
        match = re.search(r"^\s*REFLECTION:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        reflection = match.group(1) if match else response_text
        return " ".join(reflection.split()[:40])

    @staticmethod
    def _extract_plan_and_action(response_text):
        plan_match = re.search(
            r"^\s*PLAN:\s*(.*?)(?=^\s*ACTION:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        action_match = re.search(r"^\s*ACTION:\s*([^\r\n]+)", response_text, re.IGNORECASE | re.MULTILINE)
        plan = plan_match.group(1).strip() if plan_match else None
        action = action_match.group(1).strip().strip("`*'\"") if action_match else ""
        return plan, action
