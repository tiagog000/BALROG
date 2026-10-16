# Exercise 2: lightweight planning agent

## Design

The Exercise 2 agent maintains one short plan of at most three subgoals, reuses BALROG's bounded interaction history, and receives the actual `Step k of N` budget. It returns a compact `PLAN`/`ACTION` response and deliberately excludes summaries, failure ledgers, loop detectors, forced replanning, and self-reflection.

The research question is whether a small amount of persistent planning and time awareness improves multi-step continuity and reduces repeated or malformed actions relative to the naive agent, without adding the reflective mechanisms intended for Exercise 3.

## Why this is a useful midpoint

The naive trajectories show capabilities as well as limitations. The local model can follow simple navigation instructions and sometimes makes partial progress in TextWorld, but it is less reliable when a task requires preserving a multi-step intention, changing subgoals, or recovering from repeated actions. A short explicit plan gives the model a stable statement of intent while leaving environment reasoning to the model and action semantics to the existing BALROG harness.

The agent adds only three mechanisms:

1. one current plan, replaced whenever the model returns an updated plan;
2. `Step k of N`, so the model can account for the remaining budget; and
3. a strict first-line `ACTION` parser, so reasoning is not submitted as a game command.

## Fixed prompt probes

Before running the full benchmark, five prompt variants were tested against the local `gemma-4-e4b-it-mlx` model on four fixed situations. These are single-response diagnostic probes, not performance estimates.

| Situation | What the lightweight prompt showed |
|---|---|
| BabyAI subgoal transition | It updated the plan toward the remaining yellow target and selected `go forward`. |
| BabyAI locked-door approach | It returned the valid action `go forward`; this isolated case did not show a clear advantage over naive. |
| TextWorld room reversal | Naive and CoT immediately reversed east, while the stock and lightweight planning prompts selected the unexplored north exit. |
| TextWorld cooking state | It returned the valid but cautious command `look`, showing format compliance but not direct task progress. |

Under the probe's 256-token diagnostic budget, the lightweight prompt produced a parseable action in all four cases and averaged 29.5 output tokens. The stock Custom prompt produced parseable actions in three of four cases and averaged 74.25 tokens; CoT produced one of four within the budget and averaged 211 tokens. These observations motivated prompt simplicity, but only the seeded BALROG evaluation can establish whether the agent improves progression.

## State and prompt lifecycle

At reset, the agent clears its plan, interaction history, and step context. At each step it sends:

1. the environment's task instructions;
2. BALROG's bounded recent observation/action history;
3. the current observation;
4. the current plan, or `No plan yet.`; and
5. the actual step index and episode limit.

The response contract is:

```text
PLAN:
1. <subgoal>
ACTION: <one complete valid game command, including every required argument>
```

The complete plan must contain no more than three numbered subgoals and the whole response must remain under 80 words. The parser stores the returned plan and extracts only the first labelled action line.

## Scope boundary

- There is no separate summary or inventory model; relevant recent state remains in the existing bounded history.
- There is no persistent failed-attempt list, repeated-observation detector, forced alternative action, or retrospective critique.
- There are no BabyAI- or TextWorld-specific rules in Python.
- Loop detection and self-reflection remain candidates for Exercise 3.
- Passing a hidden TextWorld objective through the environment is a harness change, not an Exercise 2 agent feature, and would require rerunning every compared agent under the same harness.

## Matched local evaluation

Run each seed in a separate output directory while retaining the naive model and generation settings:

```bash
.venv/bin/python eval.py \
  agent.type=custom \
  envs.names=babyai-textworld \
  envs.env_kwargs.seed=1001 \
  client.client_name=vllm \
  client.model_id=gemma-4-e4b-it-mlx \
  client.base_url=http://127.0.0.1:1234/v1 \
  client.generate_kwargs.temperature=0.7 \
  client.generate_kwargs.max_tokens=2048 \
  client.timeout=120 \
  client.max_retries=2 \
  eval.output_dir=results/1001_gemma4_custom
```

Repeat with seeds 1002 and 1003. Compare progression, completion, repeated actions, invalid commands, steps, token use, and seed variation against the matching naive runs. Inspect trajectories to distinguish environments in which planning helps from those in which the model still misinterprets action semantics or fails to recover.
