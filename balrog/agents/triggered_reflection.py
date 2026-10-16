import re
from collections import deque

from balrog.agents.base import BaseAgent


class TriggeredReflectionAgent(BaseAgent):
    """Plan normally and request one critique only after observable failure."""

    REFLECTION_COOLDOWN_STEPS = 3
    INFORMATION_ACTIONS = {"goal", "inventory", "look"}
    FAILURE_PATTERNS = (
        ("previous output did not contain a valid action", r"previous output did not contain a valid action"),
        ("not a verb", r"not a verb"),
        ("you can't", r"^\s*you can't\b"),
        ("you cannot", r"^\s*you cannot\b"),
        ("you have to", r"^\s*you have to\b"),
        ("doesn't seem to fit", r"^\s*that doesn't seem to fit\b"),
    )

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.plan = None
        self.current_lesson = None
        self.recent_observations = deque(maxlen=3)
        self.recent_actions = deque(maxlen=4)
        self.recent_attempts = deque(maxlen=16)
        self.failed_attempts = deque(maxlen=3)
        self.last_observation_signature = None
        self.last_reflection_step = None
        self.last_reflection_key = None

    def reset(self):
        """Clear all episode-specific planning and reflection state."""
        super().reset()
        self.plan = None
        self.current_lesson = None
        self.recent_observations.clear()
        self.recent_actions.clear()
        self.recent_attempts.clear()
        self.failed_attempts.clear()
        self.last_observation_signature = None
        self.last_reflection_step = None
        self.last_reflection_key = None

    def act(self, obs, prev_action=None):
        """Reflect when triggered, then update the plan and choose one action."""
        observation = self._observation_text(obs)
        observation_signature = self._normalise(observation)
        repeated_attempt = False

        if prev_action:
            self.prompt_builder.update_action(prev_action)
            action_signature = self._normalise(prev_action)
            self.recent_actions.append(action_signature)

            if self.last_observation_signature is not None:
                attempt = (self.last_observation_signature, action_signature)
                repeated_attempt = attempt in self.recent_attempts
                self.recent_attempts.append(attempt)

                failure_marker = self._failure_marker(observation)
                if (
                    observation_signature == self.last_observation_signature
                    and not self._is_information_action(action_signature)
                ):
                    self._remember_failed_attempt(prev_action, f"No state change; observation: {observation}")
                elif failure_marker:
                    self._remember_failed_attempt(prev_action, observation)

        self.prompt_builder.update_observation(obs)

        if self.recent_observations and observation_signature != self.recent_observations[-1]:
            self.last_reflection_key = None
        self.recent_observations.append(observation_signature)
        self.last_observation_signature = observation_signature

        trigger = self._reflection_trigger(observation, repeated_attempt)
        reflection_key = (trigger, observation_signature)

        critic_response = None
        reflection = None
        if trigger and reflection_key != self.last_reflection_key and self._cooldown_complete():
            critic_response = self.client.generate(self._critic_prompt(trigger))
            reflection = self._extract_reflection(critic_response.completion)
            if reflection:
                self.current_lesson = reflection
            self.last_reflection_step = self.episode_step
            self.last_reflection_key = reflection_key

        actor_response = self.client.generate(self._actor_prompt())
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
        failed_attempts = self._format_failed_attempts()
        plan = self.plan or "No plan yet."
        lesson = self.current_lesson or "None yet."
        instructions = f"""
You are a critic evaluating an agent that appears to be stuck.

Reflection trigger:
{trigger}

Current plan:
{plan}

Current lesson:
{lesson}

Recent actions:
{recent_actions}

Recent failed attempts:
{failed_attempts}

Identify one mistaken assumption or ineffective strategy and one specific change to make.
Base the diagnosis only on the available evidence. Do not choose the next action.
Keep the response under 30 words.

Return exactly:
REFLECTION: <one concise lesson>
        """.strip()
        messages[-1].content += "\n\n" + instructions
        return messages

    def _actor_prompt(self):
        messages = self.prompt_builder.get_prompt()
        step = self.episode_step or "?"
        max_steps = self.episode_max_steps if self.episode_max_steps is not None else "?"
        plan = self.plan or "No plan yet."
        lesson = self.current_lesson or "None."
        failed_attempts = self._format_failed_attempts()
        instructions = f"""
Step {step} of {max_steps}.

Current plan:
{plan}

Recovery lesson:
{lesson}

Recent failed attempts:
{failed_attempts}

Update the plan from the current observation, recent action history, recovery lesson, and failed attempts.
Do not retry the same action in the same state when it previously failed.
The complete plan must contain at most three numbered subgoals. Keep the response under 80 words.

Return exactly:
PLAN:
1. <subgoal>
ACTION: <one complete valid game command, including every required argument>
        """.strip()
        messages[-1].content += "\n\n" + instructions
        return messages

    def _reflection_trigger(self, observation, repeated_attempt=False):
        failure_marker = self._failure_marker(observation)
        if failure_marker:
            return f"explicit failure feedback: {failure_marker}"

        if len(self.recent_observations) == 3 and len(set(self.recent_observations)) == 1:
            return "the same observation occurred three times"

        if len(self.recent_actions) >= 3:
            last_three_actions = list(self.recent_actions)[-3:]
            if len(set(last_three_actions)) == 1 and last_three_actions[0] in {"turn left", "turn right"}:
                return "the same action occurred three times"

        if repeated_attempt:
            return "the agent revisited an observation-action pair"

        if len(self.recent_actions) == 4:
            first, second, third, fourth = self.recent_actions
            if first == third and second == fourth and first != second:
                return "the recent actions form an A-B-A-B oscillation"

        return None

    def _remember_failed_attempt(self, action, outcome):
        entry = (str(action).strip(), self._compact_outcome(outcome))
        for existing in list(self.failed_attempts):
            if (self._normalise(existing[0]), self._normalise(existing[1])) == (
                self._normalise(entry[0]),
                self._normalise(entry[1]),
            ):
                self.failed_attempts.remove(existing)
                break
        self.failed_attempts.append(entry)

    def _format_failed_attempts(self):
        if not self.failed_attempts:
            return "None."
        return "\n".join(f"- {action} -> {outcome}" for action, outcome in self.failed_attempts)

    @classmethod
    def _failure_marker(cls, observation):
        for marker, pattern in cls.FAILURE_PATTERNS:
            if re.search(pattern, observation, re.IGNORECASE | re.MULTILINE):
                return marker
        return None

    @staticmethod
    def _compact_outcome(outcome):
        compact = re.sub(r"\s+", " ", str(outcome)).strip()
        return compact[:160]

    @classmethod
    def _is_information_action(cls, action):
        return action in cls.INFORMATION_ACTIONS or action.startswith("look ")

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
