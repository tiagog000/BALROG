# Exercise 3: triggered reflection agent

## Idea

The Exercise 3 agent keeps the lightweight plan-and-act behaviour from Exercise 2, but asks a separate critic prompt for one short recovery lesson when observable evidence suggests that the current approach has failed. Reflection is event-driven rather than requested at every step, keeping the design and token overhead small while making the intervention easy to identify in trajectories.

The research question is whether a targeted critique improves recovery from loops and invalid commands in BabyAI and TextWorld without making otherwise successful decisions less stable.

## Triggers and control flow

On an ordinary step, the agent makes one actor call and returns a `PLAN`/`ACTION` response. It requests a critic call before the actor only when one of these triggers fires:

1. the same normalized observation has occurred three times;
2. an observation/action pair is revisited;
3. the previous four actions form an `A-B-A-B` oscillation; or
4. the environment reports an invalid or semantically unsuccessful command, including TextWorld feedback such as “not a verb,” “you can't,” “you have to,” or “doesn't seem to fit.”

The critic must identify one mistaken assumption or ineffective strategy and one specific change, without selecting the next action. The latest non-empty lesson persists for the rest of the episode or until a later critique replaces it. The agent also keeps at most three evidence-backed failed attempts as `action -> outcome`; these are recorded after explicit failure feedback or after the same action is retried from the same state without changing the observation.

The actor sees the current plan, retained lesson, and failed attempts before choosing an action. A reflected state is suppressed until the observation changes, and a short cooldown prevents rapid repeated critiques. Repeated actions are not themselves failures while observations continue to show progress; detection is based on state/action evidence rather than hard-coded action names.

## Prompt contracts

The critic response is:

```text
REFLECTION: <one concise lesson>
```

The actor receives `Step k of N`, the current plan, the retained recovery lesson, and the bounded failed-attempt list. Its response is:

```text
PLAN:
1. <subgoal>
ACTION: <one complete valid game command, including every required argument>
```

Both raw model responses are retained in the trajectory reasoning field on reflection steps, and their token counts are combined. This makes the cost and effect of the intervention observable.

## Scope boundary

- The Exercise 2 `custom` agent remains unchanged and is still the lightweight planning baseline.
- The Exercise 3 agent is selected separately with `agent.type=reflective`.
- It uses BALROG's existing bounded history and step context, plus one episode-local lesson and three failed attempts; it adds no summary ledger, inventory model, cross-episode memory, or puzzle-specific action policy.
- Python detects symptoms but does not prescribe a puzzle action. The language model diagnoses the failure and chooses the recovery.
- The implementation is limited to mechanisms relevant to BabyAI and TextWorld, although it does not hardcode task solutions for either environment.

## Matched local evaluation

Run the same local model, generation settings, episode counts, and seeds used for Exercise 2. Use a distinct output directory for each seed:

```bash
for seed in 1001 1002 1003; do
  .venv/bin/python eval.py \
    agent.type=reflective \
    envs.names=babyai-textworld \
    envs.env_kwargs.seed=$seed \
    client.client_name=vllm \
    client.model_id=gemma-4-e4b-it-mlx \
    client.base_url=http://127.0.0.1:1234/v1 \
    client.generate_kwargs.temperature=0.7 \
    client.generate_kwargs.max_tokens=2048 \
    client.timeout=120 \
    client.max_retries=2 \
    eval.num_episodes.babyai=10 \
    eval.num_episodes.textworld=10 \
    eval.num_workers=2 \
    eval.output_dir=results/${seed}_gemma4_reflective
done
```

Compare it with naive and Exercise 2 under matched settings. In addition to progression and completion, report:

- reflection triggers per episode and trigger type;
- failed attempts retained and repeated after being recorded;
- progression after a trigger versus before it;
- repeated-observation and action-oscillation rates;
- invalid-command rates;
- steps and total input/output tokens; and
- variation across environments, tasks, and seeds.

## Stability and brittleness questions

- Does the same failure pattern produce similar lessons and recoveries across seeds?
- How often does a critique break the loop, merely rename the failed strategy, or introduce a new error?
- Are false-positive triggers interrupting behaviour that was making progress?
- Does the extra critic call improve progression enough to justify its token cost?
- Are improvements concentrated in navigation loops, parser failures, or particular BabyAI/TextWorld tasks?
