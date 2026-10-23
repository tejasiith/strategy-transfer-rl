import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from env import make_source, make_target, make_hard_target, ONTOLOGY
from agent import q_learning, record_demo
from strategy import extract_strategy, generalise

N_SEEDS = 20
N_DEMOS = 30
WINDOW = 20


def with_ontology(event):
    verb, obj = event
    return (verb, ONTOLOGY.get(obj, obj))


def raw_tokens(event):
    return event


def episodes_to_threshold(row, episodes, threshold=0.9):
    avg = np.convolve(row, np.ones(WINDOW) / WINDOW, mode="valid")
    hits = np.where(avg >= threshold)[0]
    return hits[0] + WINDOW if len(hits) else episodes


def run_target(label, make_env, episodes, conditions):
    results = {}
    for name, kwargs in conditions.items():
        runs = []
        for s in range(N_SEEDS):
            kw = dict(kwargs)
            if "pick" in kw:
                kw["strategy"] = kw.pop("pick")(s)
            runs.append(q_learning(make_env(), episodes, seed=100 + s, **kw)[1])
        results[name] = np.array(runs)
        print("finished:", label, "-", name)

    print("\n%s target" % label)
    print("%-30s %-20s %-9s %-14s %s" % (
        "condition", "episodes to 90%", "reached", "first 200 eps", "last 100 eps"))
    for name, runs in results.items():
        needed = np.array([episodes_to_threshold(r, episodes) for r in runs])
        reached = int((needed < episodes).sum())
        se = needed.std() / np.sqrt(N_SEEDS)
        print("%-30s %7.1f +/- %-8.1f %2d/%-6d %-14.2f %.2f" % (
            name, needed.mean(), se, reached, N_SEEDS,
            runs[:, :200].mean(), runs[:, -100:].mean()))

    x = np.arange(WINDOW, episodes + 1)
    plt.figure(figsize=(7, 4.2))
    for name, runs in results.items():
        smooth = np.array([np.convolve(r, np.ones(WINDOW) / WINDOW, mode="valid") for r in runs])
        mean = smooth.mean(axis=0)
        se = smooth.std(axis=0) / np.sqrt(N_SEEDS)
        style = dict(ls="--", lw=2.5) if name.startswith("Q-learning") else dict(lw=1.5)
        plt.plot(x, mean, label=name, **style)
        plt.fill_between(x, mean - se, mean + se, alpha=0.2)
    plt.xlabel("training episode (target task)")
    plt.ylabel("success rate (20-episode average)")
    plt.title("%s target, %d seeds (mean +/- s.e.)" % (label, N_SEEDS))
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/curves_%s.png" % label, dpi=150)
    plt.close()


Q_source, success = q_learning(make_source(random_start=True), 1500, seed=0)
print("source success rate, last 100 episodes:", success[-100:].mean())

rng = random.Random(1)
demos = []
while len(demos) < N_DEMOS:
    demo = record_demo(make_source(random_start=True), Q_source, rng, eps=0.3)
    if demo:
        demos.append(demo)

distinct = {tuple(d) for d in demos}
print("distinct demo sequences:", len(distinct))
for d in list(distinct)[:5]:
    print(" ", list(d))

raw_strategy = extract_strategy(demos)
strategy = generalise(raw_strategy, ONTOLOGY)
print("\nextracted strategy:", raw_strategy)
print("after ontology:    ", strategy)

conditions = {
    "Q-learning (no strategy)": dict(),
    "Strategy, no ontology": dict(strategy=raw_strategy, abstract=raw_tokens),
    "Single demo + ontology": dict(
        pick=lambda s: generalise(demos[s % len(demos)], ONTOLOGY),
        abstract=with_ontology),
    "Aligned strategy + ontology": dict(strategy=strategy, abstract=with_ontology),
}

run_target("easy", make_target, 800, conditions)
run_target("hard", make_hard_target, 1500, conditions)