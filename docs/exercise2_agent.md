# Exercise 2: bounded-memory planning agent

## Two-sentence design

The Exercise 2 agent creates and maintains a short subgoal plan while replacing its memory at every step with one compact summary of the objective, completed and current subgoals, game state, and failed approaches. It receives the real `Step k of N` budget and uses tolerant `SUMMARY`/`PLAN`/`ACTION` extraction, aiming to preserve useful state without replaying a growing observation or plan history.

## Evidence from the naive baseline

The design targets observed failures rather than assuming that more reasoning is always beneficial.

| Observed failure | Trajectory evidence | Agent mechanism |
|---|---|---|
| Repeats an action after the observation stops changing | Seed 1001, BabyAI `putnext` run 00 executes `go forward` 120 times in 128 steps, often while facing the same wall; progression is 0. | Record unchanged-observation attempts in a bounded failure list and include them in the next decision. |
| Loses track of multi-stage missions | All three naive seeds score 0 on `putnext`; the example mission is “put the purple key next to the yellow key” but never reaches object placement. | Maintain ordered subgoals, their completion state, and exactly one current subgoal. |
| Repeats an interaction without updating object state | Seed 1001, BabyAI `open` run 04 carries a purple key but uses `toggle` 47 times and `go forward` 77 times without opening the target door. | Require the latest summary to retain inventory and object state, and revise the plan when the observation contradicts it. |
| Oscillates between explored rooms | Seed 1001, TextWorld `coin_collector` run 00 alternates `go east` and `go west` for all 80 steps. | Retain visited locations, exits, and failed routes in the latest summary. Active code-forced loop breaking is reserved for Exercise 3. |
| Produces commands that express reasoning rather than valid game actions | Seed 1001, `the_cooking_game` run 00 submits a multi-paragraph explanation ending in `go south`; TextWorld replies that it is not a recognised verb. | Parse one labelled action while keeping the complete raw response in the reasoning log for audit. |
| Forgets state while making partial progress | The same cooking run collects cheese and tuna and reaches 23.53% progression, but then repeats `goal`, revisits rooms, and mixes bare directions with canonical commands. | Summarise location, inventory, transformations, tried approaches, and the next intended subgoal in a fixed-size state. |
| Ignores the available budget | BabyAI examples consume all 128 steps and TextWorld examples consume all 80 steps despite long unproductive loops. | Add the evaluator’s actual `Step k of N` to every prompt. |

The inspected JSON and CSV files are under `results/1001_gemma4_naive/`. The same aggregate weaknesses appear across the three completed naive seeds: 0 progression on BabyAI `putnext` and TextWorld `coin_collector`, plus high seed variation on the remaining tasks.

## State and prompt lifecycle

At reset, the agent clears its objective, plan, summary, failed attempts, and step context. At each step it sends only:

1. the benchmark task instructions;
2. the current objective and observation;
3. the latest summary;
4. the complete current plan;
5. a bounded list of immediately detected failed attempts; and
6. the actual step index and episode limit.

The response contract is:

```text
SUMMARY: OBJECTIVE: ... | SUBGOALS: ... | STATE: ... | TRIED_AND_FAILED: ... | NEXT: ...
PLAN: <complete current plan>
ACTION: <one action>
```

The newly generated summary replaces the old summary. The plan is also replaced when revised; old plans and raw trajectory history are not replayed.

## Scope boundary

- The core agent contains no BabyAI- or TextWorld-specific action rules.
- Detecting an unchanged observation records evidence but does not force a different action.
- Multi-state loop detection, forced replanning, and retrospective self-critique remain Exercise 3 candidates.
- Passing TextWorld’s hidden instance objective through the environment is a harness change. It must be evaluated by rerunning every compared agent under the same harness and must not be presented as an Exercise 2 agent improvement.

## Matched local evaluation

Run each seed as a separate output directory while retaining the naive generation settings:

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

Repeat with seeds 1002 and 1003. Compare environment and task progression, completion, seed variation, repeated actions, invalid commands, steps, and token use against the matching naive runs.
