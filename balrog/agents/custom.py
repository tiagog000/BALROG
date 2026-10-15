import copy
import re
from collections import deque

from balrog.agents.base import BaseAgent
from balrog.prompt_builder.history import Message


class CustomAgent(BaseAgent):
    """A bounded-memory planning agent for long-horizon game tasks."""

    DEFAULT_MAX_SUMMARY_CHARS = 1800
    DEFAULT_MAX_PLAN_CHARS = 1200
    DEFAULT_MAX_FAILED_ATTEMPTS = 6

    def __init__(self, client_factory, prompt_builder, config=None):
        super().__init__(client_factory, prompt_builder)
        self.max_summary_chars = self._config_value(
            config, "custom_max_summary_chars", self.DEFAULT_MAX_SUMMARY_CHARS
        )
        self.max_plan_chars = self._config_value(config, "custom_max_plan_chars", self.DEFAULT_MAX_PLAN_CHARS)
        self.max_failed_attempts = self._config_value(
            config, "custom_max_failed_attempts", self.DEFAULT_MAX_FAILED_ATTEMPTS
        )
        self.failed_attempts = deque(maxlen=self.max_failed_attempts)
        self.plan = None
        self.latest_summary = None
        self.objective = None
        self._last_observation_signature = None

    @staticmethod
    def _config_value(config, key, default):
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def reset(self):
        """Clear all episode-specific planning and memory state."""
        super().reset()
        self.plan = None
        self.latest_summary = None
        self.objective = None
        self.failed_attempts.clear()
        self._last_observation_signature = None

    def act(self, obs, prev_action=None):
        """Update bounded state, revise the plan, and select one action."""
        observation = self._observation_text(obs)
        self._update_objective(obs)
        self._record_unchanged_observation(observation, prev_action)

        messages = self._build_messages(obs, observation)
        response = self.client.generate(messages)
        summary, plan, action = self._extract_sections(response.completion)

        if summary:
            self.latest_summary = self._bounded(summary, self.max_summary_chars)
        if plan and not self._means_no_plan_change(plan):
            self.plan = self._bounded(plan, self.max_plan_chars)

        self._last_observation_signature = self._normalise_observation(observation)

        final_answer = copy.deepcopy(response)
        final_answer = final_answer._replace(reasoning=response.completion, completion=action)
        return final_answer

    def _build_messages(self, obs, observation):
        messages = []
        if self.prompt_builder.system_prompt:
            messages.append(Message(role="user", content=self.prompt_builder.system_prompt))

        step = self.episode_step if self.episode_step else "?"
        max_steps = self.episode_max_steps if self.episode_max_steps is not None else "?"
        objective = self.objective or "Use the objective in the task instructions and current observation."
        summary = self.latest_summary or "No summary exists yet. Build one from the current evidence."
        plan = self.plan or "No plan exists yet. Create a short ordered plan before choosing the action."
        failed_attempts = self._format_failed_attempts()

        prompt = f"""
Step {step} of {max_steps}

Current objective:
{objective}

Current observation:
{observation}

Most recent summary (this is the only previous summary available):
{summary}

Current plan:
{plan}

Harness-detected failed attempts:
{failed_attempts}

Update the compact summary and plan using only information relevant to the current objective.
- If no plan exists, create one as ordered subgoals.
- Mark completed subgoals and identify exactly one current subgoal.
- In STATE, retain task-relevant game state such as location, inventory, visited locations and exits,
  object status, and completed transformations.
- In TRIED_AND_FAILED, retain approaches that failed and do not immediately retry them unchanged.
- Revise the plan when the observation contradicts it; otherwise keep it concise and stable.
- Respect the remaining step budget.
- Choose exactly one action permitted by the task instructions.

Return exactly these three labelled sections:
SUMMARY: OBJECTIVE: ... | SUBGOALS: ... | STATE: ... | TRIED_AND_FAILED: ... | NEXT: ...
PLAN: <the complete current plan, not the history of old plans>
ACTION: <one action and no explanation>
        """.strip()

        attachment = obs.get("image") if isinstance(obs, dict) else None
        messages.append(Message(role="user", content=prompt, attachment=attachment))
        return messages

    def _update_objective(self, obs):
        mission = obs.get("mission") if isinstance(obs, dict) else None
        if mission:
            self.objective = str(mission).strip()

    def _record_unchanged_observation(self, observation, prev_action):
        if not prev_action or self._last_observation_signature is None:
            return

        current_signature = self._normalise_observation(observation)
        if current_signature != self._last_observation_signature:
            return

        failure = f"'{str(prev_action).strip()}' left the visible observation unchanged"
        if failure not in self.failed_attempts:
            self.failed_attempts.append(failure)

    def _format_failed_attempts(self):
        if not self.failed_attempts:
            return "None recorded by the harness."
        return "\n".join(f"- {attempt}" for attempt in self.failed_attempts)

    @staticmethod
    def _observation_text(obs):
        if not isinstance(obs, dict):
            return str(obs)

        text = obs.get("text", {})
        if not isinstance(text, dict):
            return str(text)

        parts = [text.get("short_term_context", ""), text.get("long_term_context", "")]
        rendered = "\n".join(str(part).strip() for part in parts if part and str(part).strip())
        return rendered or "No textual observation was provided."

    @staticmethod
    def _normalise_observation(observation):
        return re.sub(r"\s+", " ", observation).strip().lower()

    @staticmethod
    def _means_no_plan_change(plan):
        normalised = re.sub(r"[^a-z]+", " ", plan.lower()).strip()
        return normalised in {
            "no change",
            "no changes",
            "no changes to the plan",
            "keep current plan",
            "keep the current plan",
        }

    @staticmethod
    def _bounded(value, limit):
        value = value.strip()
        if len(value) <= limit:
            return value
        return value[: limit - 3].rstrip() + "..."

    @staticmethod
    def _extract_sections(response_text):
        """Tolerantly extract the latest summary, complete plan, and one action."""
        summary_match = re.search(
            r"^\s*SUMMARY:\s*(.*?)(?=^\s*PLAN:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        plan_match = re.search(
            r"^\s*PLAN:\s*(.*?)(?=^\s*ACTION:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        action_match = re.search(r"^\s*ACTION:\s*([^\r\n]+)", response_text, re.IGNORECASE | re.MULTILINE)

        summary = summary_match.group(1).strip() if summary_match else None
        plan = plan_match.group(1).strip() if plan_match else None

        if action_match:
            action = action_match.group(1).strip()
        else:
            tagged_action = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", response_text, re.DOTALL)
            if tagged_action:
                action = tagged_action.group(1).strip()
            else:
                nonempty_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
                action = nonempty_lines[-1] if nonempty_lines else ""

        return summary, plan, CustomAgent._clean_action(action)

    @staticmethod
    def _clean_action(action):
        return action.strip().strip("`*'\"").strip()
