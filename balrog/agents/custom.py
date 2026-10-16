import re

from balrog.agents.base import BaseAgent


class CustomAgent(BaseAgent):
    """A lightweight planner that keeps one short plan across environment steps."""

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.plan = None

    def reset(self):
        """Clear the plan and BALROG's bounded interaction history."""
        super().reset()
        self.plan = None

    def act(self, obs, prev_action=None):
        """Update the current plan and choose one action."""
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        step = self.episode_step or "?"
        max_steps = self.episode_max_steps if self.episode_max_steps is not None else "?"
        plan = self.plan or "No plan yet."
        planning_instructions = f"""
Step {step} of {max_steps}.

Current plan:
{plan}

Update the plan from the current observation and recent action history. The complete plan must
contain at most three numbered subgoals; combine related actions into one subgoal. Keep the entire
response under 80 words.

Return exactly:
PLAN:
1. <subgoal>
ACTION: <one complete valid game command, including every required argument>
        """.strip()

        messages = self.prompt_builder.get_prompt()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + planning_instructions

        response = self.client.generate(messages)
        plan, action = self._extract_plan_and_action(response.completion)
        if plan:
            self.plan = plan

        return response._replace(reasoning=plan or self.plan, completion=action)

    @staticmethod
    def _extract_plan_and_action(response_text):
        """Extract a complete plan and only the first labelled action line."""
        plan_match = re.search(
            r"^\s*PLAN:\s*(.*?)(?=^\s*ACTION:|\Z)",
            response_text,
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        action_match = re.search(r"^\s*ACTION:\s*([^\r\n]+)", response_text, re.IGNORECASE | re.MULTILINE)

        plan = plan_match.group(1).strip() if plan_match else None
        action = action_match.group(1).strip().strip("`*'\"") if action_match else ""

        return plan, action
