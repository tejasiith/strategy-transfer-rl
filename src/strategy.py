def align(a, b, match=2, mismatch=-1, gap=-1):
    n, m = len(a), len(b)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
    for j in range(1, m + 1):
        score[0][j] = j * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            score[i][j] = max(diag, score[i - 1][j] + gap, score[i][j - 1] + gap)

    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        diag = score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
        if score[i][j] == diag:
            if a[i - 1] == b[j - 1]:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def extract_strategy(demos, min_support=0.6):
    # reference = the demo that lines up best with all the others
    def total_matches(i):
        return sum(len(align(demos[i], other)) for other in demos)

    ref = max(range(len(demos)), key=total_matches)
    support = [1] * len(demos[ref])
    for k, other in enumerate(demos):
        if k == ref:
            continue
        for i, _ in align(demos[ref], other):
            support[i] += 1

    return [tok for tok, s in zip(demos[ref], support)
            if s / len(demos) >= min_support]


def generalise(events, ontology):
    return [(verb, ontology.get(obj, obj)) for verb, obj in events]