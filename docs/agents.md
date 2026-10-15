# Agents

In BALROG, agents are entities that typically wrap an LLM client. They are responsible for receiving observations and selecting actions. As such, agents are internally responsible for:
1. Maintaining Observation/Action Histories: Agents keep a record of past observations and actions to support context-aware decision-making.
2. Querying LLMs: Agents send observations to the LLM, receive responses, and use those responses to decide on actions.

## Pre-built agents

BALROG ships with two pre-built agents:

| **Agent Type**          | **Description**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|
| **naive**          | Outputs actions based on the current action/observation history without any additional reasoning. |
| **robust_naive**          | Outputs actions based on the current action/observation history without any additional reasoning using a more robust template. |
| **chain_of_thought** | Generates actions through step-by-step reasoning, providing a final action output. |


We encourage the community to open PRs to include more agents to BALROG.

## 🤖 Creating Custom Agents

The simple zero-shot agent in `naive.py` outputs only a single action with no extra reasoning, but this is often suboptimal. We may want the agent to analyze its situation, form and refine plans, interpret image observations, or handle history more effectively.

To build a custom agent, you’ll mainly work with:
1. `balrog/agents/custom.py` -> your custom agent file.
2. `balrog/prompt_builder/history.py` -> containing the history prompt builder, an helper class to deal with with observation/action history in prompts.

You’re free to modify or create additional files, as long as they don’t interfere with evaluation, logging, or environment processes.


### Bounded-memory planning agent

The repository's `custom` agent maintains one replaceable summary, one current subgoal plan, a bounded list of failed attempts, and the actual episode step budget. It deliberately avoids replaying the raw observation history or obsolete plans. See [Exercise 2: bounded-memory planning agent](exercise2_agent.md) for its design rationale, naive-baseline evidence, response contract, and matched evaluation command.


You can then evaluate the custom agent with:
```
.venv/bin/python eval.py \
  agent.type=custom \
  agent.max_image_history=0 \
  eval.num_workers=64 \
  client.client_name=openai \
  client.model_id=gpt-4o-mini-2024-07-18
```

Check our [evaluation tutorial](https://github.com/balrog-ai/BALROG/blob/main/docs/evaluation.md) to learn more on how to evaluate your agents using BALROG.
