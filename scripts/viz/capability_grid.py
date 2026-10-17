#!/usr/bin/env python3
"""Defragmenter-style capability grids for the BALROG eval matrix.

Reads every results/1610_1145-<seed>_<model>_<agent>/ run, extracts per-episode
records, and renders several complementary views of "which subtasks each
model/agent/seed could solve", with progression as the z (colour) channel.

Run:  .venv/bin/python scripts/viz/capability_grid.py
Out:  results/viz/*.png
"""
from __future__ import annotations
import json, re, glob, os, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


def header(fig, question, look_for, top=None):
    """Stamp a bold QUESTION + an italic 'what to look for' line.

    Offsets are in INCHES (converted to figure-fraction via the figure height)
    so the title and subtitle keep a fixed gap and never collide on the shorter
    figures. Returns the rect-top to reserve for the header band so callers can
    pass it straight to tight_layout(rect=[0, 0, 1, <returned>]).
    """
    fh = fig.get_figheight()
    fig.suptitle("Q:  " + question, fontsize=13, fontweight="bold", y=1 - 0.30 / fh)
    wrapped = textwrap.fill("What to look for:  " + look_for, width=100)
    nlines = wrapped.count("\n") + 1
    fig.text(0.5, 1 - 0.58 / fh, wrapped, ha="center", va="top",
             fontsize=9.5, style="italic", color="#333333")
    # reserve the header band: title + each wrapped subtitle line (~0.22in/line)
    return 1 - (0.80 + 0.22 * (nlines - 1)) / fh

ROOT = Path(__file__).resolve().parents[2]          # BALROG/
RES = ROOT / "results"
OUT = RES / "viz"
OUT.mkdir(parents=True, exist_ok=True)

BABYAI = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
TEXTWORLD = ["treasure_hunter", "the_cooking_game", "coin_collector"]
TASKS = BABYAI + TEXTWORLD
TASK_LABEL = {"goto": "goto", "pickup": "pickup", "open": "open", "putnext": "putnext",
              "pick_up_seq_go_to": "pickseq", "treasure_hunter": "treasure",
              "the_cooking_game": "cooking", "coin_collector": "coin"}
# agent display order (capability ladder) + nice labels
AGENT_ORDER = ["naive", "custom", "reflective", "robust_cot",
               "reflective_robust_cot", "reflective_robust_cot_fix",
               "reflective_robust_cot_free"]
AGENT_LABEL = {"naive": "naive", "custom": "custom", "reflective": "reflective",
               "robust_cot": "robust_cot", "reflective_robust_cot": "hybrid",
               "reflective_robust_cot_fix": "fix", "reflective_robust_cot_free": "free"}

# user focus: the three agents under comparison (keeps charts legible)
FOCUS = ["naive", "custom", "reflective_robust_cot_free"]


def focus_agents(rows):
    """Agents to show, in ladder order, restricted to the FOCUS set present."""
    return [a for a in AGENT_ORDER if a in FOCUS and any(r["agent"] == a for r in rows)]

# defrag palette: red (unsolved) -> amber (partial) -> green (solved)
CMAP = LinearSegmentedColormap.from_list(
    "defrag", ["#8b0000", "#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"])
CMAP.set_bad("#2b2b2b")   # NaN = not run (dark grey)


def episode_files(run_dir: Path, env: str, task: str):
    if env == "babyai":
        pats = [run_dir / "*" / "babyai" / "*" / task / "*" / f"{task}_run_*.json",
                run_dir / "*" / "babyai" / "*" / task / f"{task}_run_*.json"]
    else:
        pats = [run_dir / "*" / "textworld" / task / f"{task}_run_*.json"]
    files = []
    for p in pats:
        files += glob.glob(str(p))
    return sorted(set(files))


def load():
    rows = []
    for d in sorted(glob.glob(str(RES / "1610_1145-*_*"))):
        m = re.search(r"1610_1145-(\d+)_([a-z0-9]+)_(.+)$", os.path.basename(d))
        if not m:
            continue
        seed, model, agent = int(m.group(1)), m.group(2), m.group(3)
        if model not in ("qwen", "gemma4"):
            continue
        for env, tasklist in (("babyai", BABYAI), ("textworld", TEXTWORLD)):
            for task in tasklist:
                for i, f in enumerate(episode_files(Path(d), env, task)):
                    try:
                        js = json.load(open(f))
                    except Exception:
                        continue
                    af = js.get("action_frequency") or {}
                    tot = sum(af.values()) or 0
                    gf = af.get("go forward", 0)
                    rows.append(dict(
                        seed=seed, model=model, agent=agent, env=env, task=task,
                        run=i, progression=js.get("progression"),
                        ret=js.get("episode_return"), steps=js.get("num_steps"),
                        done=bool(js.get("done")), mission=js.get("mission"),
                        out_tok=js.get("output_tokens") or 0,
                        gf_frac=(gf / tot) if tot else np.nan))
    return rows


def run_key(model, agent, seed):
    return (0 if model == "qwen" else 1,
            AGENT_ORDER.index(agent) if agent in AGENT_ORDER else 99, seed)


# ----------------------------------------------------------------------------- #
def fig_defrag(rows):
    """Per-episode defrag grid: one row per run, 80 episode cells in task order."""
    runs = sorted({(r["model"], r["agent"], r["seed"]) for r in rows if r["agent"] in FOCUS},
                  key=lambda t: run_key(*t))
    # fixed episode columns: 10 per task => 80 columns
    PER = 10
    cols = len(TASKS) * PER
    grid = np.full((len(runs), cols), np.nan)
    for r in rows:
        if r["progression"] is None or r["agent"] not in FOCUS:
            continue
        ri = runs.index((r["model"], r["agent"], r["seed"]))
        ci = TASKS.index(r["task"]) * PER + min(r["run"], PER - 1)
        grid[ri, ci] = r["progression"]

    fig, ax = plt.subplots(figsize=(14, 0.5 * len(runs) + 2))
    ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{AGENT_LABEL.get(a,a)}@{s}" + ("" if m == "qwen" else "  (gemma)")
                        for m, a, s in runs], fontsize=8)
    # task boundaries + labels
    ax.set_xticks([t * PER + PER / 2 - 0.5 for t in range(len(TASKS))])
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], fontsize=8)
    for t in range(1, len(TASKS)):
        ax.axvline(t * PER - 0.5, color="white", lw=0.6)
    ax.axvline(len(BABYAI) * PER - 0.5, color="cyan", lw=1.8)   # babyai|textworld divide
    # model band separator
    first_gemma = next((i for i, (m, _, _) in enumerate(runs) if m == "gemma4"), None)
    if first_gemma:
        ax.axhline(first_gemma - 0.5, color="white", lw=1.5)
    ax.text(len(BABYAI) * PER / 2, -0.9, "BabyAI (embodied nav)", ha="center", fontsize=9, color="#1a9641")
    ax.text(len(BABYAI) * PER + len(TEXTWORLD) * PER / 2, -0.9, "TextWorld (symbolic)",
            ha="center", fontsize=9, color="#2b8cbe")
    ax.set_title("each cell = 1 episode · colour = progression (0 red → 1 green) · grey = not run",
                 fontsize=9, color="#333333")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("progression")
    rtop = header(fig,
           "Which subtasks can each agent actually solve — episode by episode, per seed?",
           "green runs of cells = reliable solves; red = never; a row that is red on BabyAI "
           "but not on TextWorld (or vice-versa) shows where a harness helps vs hurts.")
    fig.tight_layout(rect=[0, 0, 1, rtop])
    fig.savefig(OUT / "capability_defrag_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap_by_seed(rows):
    """Agent × subtask mean-progression heatmap, one facet per seed (Qwen)."""
    qrows = [r for r in rows if r["model"] == "qwen"]
    seeds = sorted({r["seed"] for r in qrows})
    agents = focus_agents(qrows)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 0.5 * len(agents) + 3.0),
                             squeeze=False)
    for si, seed in enumerate(seeds):
        ax = axes[0][si]
        M = np.full((len(agents), len(TASKS)), np.nan)
        for ai, ag in enumerate(agents):
            for ti, task in enumerate(TASKS):
                vals = [r["progression"] for r in qrows
                        if r["seed"] == seed and r["agent"] == ag and r["task"] == task
                        and r["progression"] is not None]
                if vals:
                    M[ai, ti] = 100 * np.mean(vals)
        ax.imshow(M, aspect="auto", cmap=CMAP, vmin=0, vmax=100)
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels([AGENT_LABEL.get(a, a) for a in agents], fontsize=8)
        ax.axvline(len(BABYAI) - 0.5, color="cyan", lw=1.5)
        for ai in range(len(agents)):
            for ti in range(len(TASKS)):
                v = M[ai, ti]
                if not np.isnan(v):
                    ax.text(ti, ai, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="black" if 20 < v < 80 else "white")
        ax.set_title(f"seed {seed}", fontsize=10)
    rtop = header(fig,
           "How much does capability depend on agent × subtask × seed?  (Qwen, mean progression %)",
           "read down a column for subtask difficulty, across a row for the agent's reach, "
           "and compare the 3 facets for seed sensitivity — numbers are % progression; "
           "left of cyan = BabyAI, right = TextWorld.")
    fig.tight_layout(rect=[0, 0, 1, rtop])
    fig.savefig(OUT / "agent_subtask_heatmap_by_seed.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_extra(rows):
    """Three extra dimensions: spin signature, cost-vs-capability, seed difficulty."""
    qrows = [r for r in rows if r["model"] == "qwen"]
    agents = focus_agents(qrows)
    seeds = sorted({r["seed"] for r in qrows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # (a) spin signature: mean go_forward fraction (BabyAI only) agent × seed
    ax = axes[0]
    S = np.full((len(agents), len(seeds)), np.nan)
    for ai, ag in enumerate(agents):
        for si, seed in enumerate(seeds):
            vals = [r["gf_frac"] for r in qrows if r["agent"] == ag and r["seed"] == seed
                    and r["env"] == "babyai" and not np.isnan(r["gf_frac"])]
            if vals:
                S[ai, si] = 100 * np.mean(vals)
    im = ax.imshow(S, aspect="auto", cmap="viridis", vmin=0, vmax=40)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds, fontsize=8)
    ax.set_yticks(range(len(agents))); ax.set_yticklabels([AGENT_LABEL.get(a, a) for a in agents], fontsize=8)
    for ai in range(len(agents)):
        for si in range(len(seeds)):
            if not np.isnan(S[ai, si]):
                ax.text(si, ai, f"{S[ai,si]:.0f}%", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Q: does the agent ever MOVE?\n% of actions that are 'go forward' (BabyAI)\ndark/low = rotation-loop collapse (spin)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) cost vs capability: output tokens/ep vs BabyAI progression, point per (agent,seed)
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
    for ai, ag in enumerate(agents):
        for seed in seeds:
            er = [r for r in qrows if r["agent"] == ag and r["seed"] == seed and r["env"] == "babyai"
                  and r["progression"] is not None]
            if not er:
                continue
            x = np.mean([r["out_tok"] for r in er])
            y = 100 * np.mean([r["progression"] for r in er])
            ax.scatter(x, y, color=colors[ai], s=60, edgecolor="k", lw=0.5,
                       label=AGENT_LABEL.get(ag, ag) if seed == seeds[0] else None)
            ax.annotate(str(seed)[-1], (x, y), fontsize=6, ha="center", va="center")
    ax.set_xlabel("mean output tokens / episode"); ax.set_ylabel("BabyAI progression (%)")
    ax.set_title("Q: does spending more tokens buy capability?\n(point label = seed; up-left = efficient)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(alpha=0.3)

    # (c) seed difficulty: BabyAI mean per agent, grouped bars by seed
    ax = axes[2]
    width = 0.8 / len(seeds)
    for si, seed in enumerate(seeds):
        ys = []
        for ag in agents:
            vals = [r["progression"] for r in qrows if r["agent"] == ag and r["seed"] == seed
                    and r["env"] == "babyai" and r["progression"] is not None]
            ys.append(100 * np.mean(vals) if vals else 0)
        ax.bar(np.arange(len(agents)) + si * width, ys, width, label=f"seed {seed}")
    ax.set_xticks(np.arange(len(agents)) + width * (len(seeds) - 1) / 2)
    ax.set_xticklabels([AGENT_LABEL.get(a, a) for a in agents], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("BabyAI progression (%)")
    ax.set_title("Q: is any seed simply unsolvable?\nBabyAI mean by agent × seed\n(1003 = hardest, but custom/free still solve it)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    rtop = header(fig,
           "WHY do agents fail — is it motion, cost, or the seed?  (three lenses on the same Qwen runs)",
           "left: a 0% with near-zero go-forward is a spin, not a hard task · "
           "middle: points far right but low = tokens spent without payoff · "
           "right: no seed is a flat zero for every agent, so none is truly unsolvable.")
    fig.tight_layout(rect=[0, 0, 1, rtop])
    fig.savefig(OUT / "extra_dimensions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _mission_for(rows, seed, agent, task):
    ms = [r.get("mission") for r in rows if r["seed"] == seed and r["agent"] == agent
          and r["task"] == task and r.get("mission")]
    return ms[0] if ms else ""


def fig_missions(rows):
    """Per-seed BabyAI: each subtask is ONE repeated mission; show progression
    (annotation) coloured by go-forward fraction, so capability and the spin
    mechanism appear together. Answers 'why is seed 1003 hard?'."""
    qrows = [r for r in rows if r["model"] == "qwen" and r["env"] == "babyai"]
    seeds = sorted({r["seed"] for r in qrows})
    agents = focus_agents(qrows)
    fig, axes = plt.subplots(len(seeds), 1,
                             figsize=(1.4 * len(agents) + 5, 2.7 * len(seeds)), squeeze=False)
    for si, seed in enumerate(seeds):
        ax = axes[si][0]
        GF = np.full((len(BABYAI), len(agents)), np.nan)   # colour = go-forward %
        PR = np.full((len(BABYAI), len(agents)), np.nan)   # text  = progression %
        ylabels = []
        for ti, task in enumerate(BABYAI):
            # mission is seed-fixed; grab from first agent that has it
            mission = next((_mission_for(qrows, seed, a, task) for a in agents
                            if _mission_for(qrows, seed, a, task)), task)
            ylabels.append(f"{TASK_LABEL[task]}\n“{mission}”")
            for ai, ag in enumerate(agents):
                er = [r for r in qrows if r["seed"] == seed and r["agent"] == ag and r["task"] == task
                      and r["progression"] is not None]
                if er:
                    PR[ti, ai] = 100 * np.mean([r["progression"] for r in er])
                    gf = [r["gf_frac"] for r in er if not np.isnan(r["gf_frac"])]
                    if gf:
                        GF[ti, ai] = 100 * np.mean(gf)
        # colour each cell by its OUTCOME category (same taxonomy as failure_modes),
        # so one glance answers "solved / partial / wandered-but-failed / spun-in-place".
        # text stays the exact progression %.
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cat_idx = {c: i for i, c in enumerate(FM_ORDER)}
        cmap = ListedColormap([FM_COLORS[c] for c in FM_ORDER]); cmap.set_bad("#2b2b2b")
        norm = BoundaryNorm(range(len(FM_ORDER) + 1), cmap.N)
        CAT = np.full((len(BABYAI), len(agents)), np.nan)
        for ti in range(len(BABYAI)):
            for ai in range(len(agents)):
                pr, gf = PR[ti, ai], GF[ti, ai]
                if np.isnan(pr):
                    continue
                if pr >= 99:              cat = "solved"
                elif pr > 0:              cat = "partial"
                elif not np.isnan(gf) and gf < 100 * SPIN_GF:  cat = "spin"
                else:                     cat = "wander"
                CAT[ti, ai] = cat_idx[cat]
        ax.imshow(CAT, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(agents))); ax.set_xticklabels([AGENT_LABEL.get(a, a) for a in agents], fontsize=8)
        ax.set_yticks(range(len(BABYAI))); ax.set_yticklabels(ylabels, fontsize=7)
        for ti in range(len(BABYAI)):
            for ai in range(len(agents)):
                if not np.isnan(PR[ti, ai]):
                    solved = PR[ti, ai] >= 99
                    # black text only on the light amber (partial); white on the rest
                    txtcol = "black" if CAT[ti, ai] == cat_idx["partial"] else "white"
                    ax.text(ai, ti, f"{PR[ti,ai]:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold" if solved else "normal", color=txtcol)
        ax.set_title(f"seed {seed}", fontsize=10)
    # one shared legend for the outcome colours, centered under the figure
    handles = [Patch(facecolor=FM_COLORS[c], label=c) for c in FM_ORDER]
    fig.legend(handles, FM_ORDER, ncol=4, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    rtop = header(fig,
           "Is a 0% because the MISSION is impossible, or because the agent won't move?",
           "each subtask is ONE fixed mission (printed on the row); the number is progression %, "
           "the colour is the outcome. A purple (spin) cell next to any non-purple cell on the "
           "SAME mission proves the task is reachable and the agent just froze — e.g. seed-1003 goto "
           "“go to the green key”: naive spun (purple 0%), while custom/free actually moved (amber).")
    fig.tight_layout(rect=[0, 0.05, 1, rtop])
    fig.savefig(OUT / "mission_breakdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


FM_COLORS = {"solved": "#1a9641", "partial": "#fdae61", "wander": "#2c7fb8", "spin": "#7b3294"}
FM_ORDER = ["solved", "partial", "wander", "spin"]
SPIN_GF = 0.08   # <8% of actions are 'go forward' => rotation-loop


def classify(r):
    p = r["progression"]
    if p is None:
        return None
    if p >= 0.999:
        return "solved"
    if p > 0:
        return "partial"
    gf = r["gf_frac"]
    if not np.isnan(gf) and gf < SPIN_GF:
        return "spin"
    return "wander"


def fig_failuremodes(rows):
    """Taxonomy of BabyAI episode outcomes: solved / partial / wander / spin.
    Left: categorical defrag grid (per-episode). Right: stacked composition per run."""
    qrows = [r for r in rows if r["model"] == "qwen" and r["env"] == "babyai"]
    runs = sorted({(r["agent"], r["seed"]) for r in qrows if r["agent"] in FOCUS},
                  key=lambda t: (AGENT_ORDER.index(t[0]) if t[0] in AGENT_ORDER else 99, t[1]))
    PER = 10
    cols = len(BABYAI) * PER
    cat_idx = {c: i for i, c in enumerate(FM_ORDER)}
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([FM_COLORS[c] for c in FM_ORDER]); cmap.set_bad("#2b2b2b")
    norm = BoundaryNorm(range(len(FM_ORDER) + 1), cmap.N)

    grid = np.full((len(runs), cols), np.nan)
    comp = {r: {c: 0 for c in FM_ORDER} for r in runs}
    for r in qrows:
        c = classify(r)
        if c is None or r["agent"] not in FOCUS:
            continue
        ri = runs.index((r["agent"], r["seed"]))
        ci = BABYAI.index(r["task"]) * PER + min(r["run"], PER - 1)
        grid[ri, ci] = cat_idx[c]
        comp[(r["agent"], r["seed"])][c] += 1

    fig, (axg, axb) = plt.subplots(1, 2, figsize=(16, 0.52 * len(runs) + 2.5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    axg.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    axg.set_yticks(range(len(runs)))
    axg.set_yticklabels([f"{AGENT_LABEL.get(a,a)}@{s}" for a, s in runs], fontsize=8)
    axg.set_xticks([t * PER + PER / 2 - 0.5 for t in range(len(BABYAI))])
    axg.set_xticklabels([TASK_LABEL[t] for t in BABYAI], fontsize=8)
    for t in range(1, len(BABYAI)):
        axg.axvline(t * PER - 0.5, color="white", lw=0.6)
    axg.set_title("per-episode outcome (BabyAI, Qwen)", fontsize=9, color="#333333")

    # stacked composition
    y = np.arange(len(runs))
    left = np.zeros(len(runs))
    for c in FM_ORDER:
        vals = np.array([comp[r][c] for r in runs], float)
        tot = np.array([sum(comp[r].values()) for r in runs], float)
        frac = np.divide(vals, tot, out=np.zeros_like(vals), where=tot > 0) * 100
        axb.barh(y, frac, left=left, color=FM_COLORS[c], label=c)
        left += frac
    axb.set_yticks(y); axb.set_yticklabels([f"{AGENT_LABEL.get(a,a)}@{s}" for a, s in runs], fontsize=8)
    # keep both panels in the SAME row order (naive@1001 bottom -> free@1003 top):
    # imshow puts row 0 at the top, barh puts it at the bottom, so only the grid flips.
    axg.invert_yaxis()
    axb.set_xlabel("% of BabyAI episodes"); axb.set_xlim(0, 100)
    axb.set_title("outcome mix per run (% of episodes)", fontsize=9, color="#333333")
    # one legend centered under the whole figure (not tucked under the right panel)
    handles, labels = axb.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    rtop = header(fig,
           "When an agent scores 0%, HOW does it fail — stuck spinning, or exploring but failing?",
           "spin (purple) = 0% & <8% go-forward (rotation-loop, a harness pathology); "
           "wander (blue) = 0% but it did advance (genuine difficulty); "
           "purple vs blue tells you whether to fix the agent or accept the task is hard.")
    fig.tight_layout(rect=[0, 0.045, 1, rtop])
    fig.savefig(OUT / "failure_modes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load()
    print(f"loaded {len(rows)} episode records across "
          f"{len({(r['model'],r['agent'],r['seed']) for r in rows})} runs")
    fig_defrag(rows)
    fig_heatmap_by_seed(rows)
    fig_extra(rows)
    fig_missions(rows)
    fig_failuremodes(rows)
    print("wrote:", *(str(p) for p in sorted(OUT.glob("*.png"))), sep="\n  ")
