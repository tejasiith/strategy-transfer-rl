import random
import numpy as np


def reached_goal(env):
    return env.objects.get(env.pos, (None, None))[1] == "objective"


def best_action(q_values, rng):
    best = np.flatnonzero(q_values == q_values.max())
    return int(rng.choice(list(best)))


class StrategyTracker:
    def __init__(self, strategy, abstract):
        self.strategy = strategy
        self.abstract = abstract
        self.stage = 0

    def reset(self):
        self.stage = 0

    def update(self, event):
        if event is None or self.stage >= len(self.strategy):
            return
        if self.abstract(event) == self.strategy[self.stage]:
            self.stage += 1

    def potential(self):
        return self.stage / len(self.strategy)


def q_learning(env, episodes, seed, strategy=None, abstract=None,
               alpha=0.2, gamma=0.99, eps=0.1):
    rng = random.Random(seed)
    Q = np.zeros((env.height, env.width, 2, 2, 4))
    tracker = StrategyTracker(strategy, abstract) if strategy else None
    success = []

    for _ in range(episodes):
        s = env.reset(rng)
        if tracker:
            tracker.reset()
        done = False

        while not done:
            if rng.random() < eps:
                a = rng.randrange(4)
            else:
                a = best_action(Q[s], rng)

            s2, r, done, event = env.step(a)
            won = done and reached_goal(env)

            # potential-based shaping: bonus for finishing strategy steps in order
            if tracker:
                before = tracker.potential()
                tracker.update(event)
                after = 0.0 if won else tracker.potential()
                r += gamma * after - before

            target = r if won else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2

        success.append(1.0 if reached_goal(env) else 0.0)

    return Q, np.array(success)


def record_demo(env, Q, rng, eps=0.15):
    s = env.reset(rng)
    events = []
    done = False
    while not done:
        if rng.random() < eps:
            a = rng.randrange(4)
        else:
            a = best_action(Q[s], rng)
        s, _, done, event = env.step(a)
        if event:
            events.append(event)
    return events if reached_goal(env) else None