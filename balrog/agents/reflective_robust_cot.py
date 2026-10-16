"""A small hybrid of triggered reflection and strict Robust CoT actions."""

import re

from balrog.agents.triggered_reflection import TriggeredReflectionAgent


class ReflectiveRobustCoTAgent(TriggeredReflectionAgent):
    """Use Robust CoT for every action and reflection only after failure.

    The parent class owns episode memory, failure detection, the critic, and
    token accounting. This subclass changes only the normal actor prompt and
    its parser, which keeps the hybrid easy to compare with both parent agents.
    """

    def _actor_prompt(self):
        """Build a planning prompt whose final action uses strict delimiters."""
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

First, think about the best course of action from the available evidence.
Before finalizing, predict its immediate effect and check that it advances the current subgoal.
Do not retry the same action in the same state when it previously failed.
Return the complete updated plan with at most three numbered subgoals.

End the response exactly as follows:
PLAN:
1. <subgoal>
<|ACTION|><one complete valid game command, including every required argument><|END|>
        """.strip()
        messages[-1].content += "\n\n" + instructions
        return messages

    def _extract_plan_and_action(self, response_text):
        """Extract a reusable plan and only a strictly delimited action.

        Free-form reasoning remains available in the recorded response but can
        never reach the environment. A missing delimiter therefore produces an
        empty action, allowing BALROG's normal invalid-action feedback to handle
        the malformed response on the following step.
        """
        plan_match = re.search(
            r"^\s*PLAN:\s*(.*?)(?=^\s*ACTION:|<\|ACTION\|>|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        action_match = re.search(
            r"<\|ACTION\|>(.*?)<\|END\|>",
            response_text,
            re.IGNORECASE | re.DOTALL,
        )
        plan = plan_match.group(1).strip() if plan_match else None
        action = action_match.group(1).strip().strip("`*'\"") if action_match else ""
        return plan, action
