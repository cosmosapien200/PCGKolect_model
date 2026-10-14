"""Beat decoder: frame posteriors -> cardiac cycles -> per-beat usability.

Hybrid HMM / Viterbi over the network's frame posteriors (50 fps):
  - cardiac states must follow S1 -> systole -> S2 -> diastole -> S1
  - each state has a minimum duration (chain of sub-states) and a geometric tail
  - a NOISE state can be entered from the end of any state and can resume into any state
Beats are S1-onset to next S1-onset spans that contain exactly one S1, systole, S2,
diastole and no noise.  A beat is USABLE when the network is confident along the decoded
path and the timing is physiologically plausible and consistent with its neighbours.

Mirrored in C++ (PCGKolect_rk/src/pcg_beats.cpp).
"""
import numpy as np

FPS = 50
NEXT = [1, 2, 3, 0]
# state:      S1  sys  S2  dia  noise
MIN_DUR = [3, 4, 3, 4, 5]        # frames
MEAN_DUR = [6, 14, 5, 25, 50]    # frames (geometric tail beyond the minimum)
P_TO_NOISE = 0.01                # from the last sub-state of a cardiac state
EPS = 1e-6

# beat acceptance (all thresholds in one place)
BEAT_MIN_S, BEAT_MAX_S = 0.30, 2.0     # 30..200 bpm
BEAT_CONF_MIN = 0.70                    # mean posterior of the decoded state along the beat
RHYTHM_TOL = 0.30                       # |beat - local median| / local median


def _build():
    """Expanded state space: list of (class, is_last_substate)."""
    states, first, last = [], [], []
    for c in range(5):
        first.append(len(states))
        for i in range(MIN_DUR[c]):
            states.append(c)
        last.append(len(states) - 1)
    n = len(states)
    logA = np.full((n, n), -np.inf)
    for c in range(5):
        for i in range(first[c], last[c]):
            logA[i, i + 1] = 0.0
        stay = 1 - 1 / max(MEAN_DUR[c] - MIN_DUR[c] + 1, 1.0001)
        l = last[c]
        if c < 4:
            logA[l, l] = np.log(stay)
            logA[l, first[NEXT[c]]] = np.log((1 - stay) * (1 - P_TO_NOISE))
            logA[l, first[4]] = np.log((1 - stay) * P_TO_NOISE)
        else:
            logA[l, l] = np.log(stay)
            for k in range(4):
                logA[l, first[k]] = np.log((1 - stay) / 4)
    return np.array(states), logA, first


STATES, LOGA, FIRST = _build()


def viterbi(P):
    """P [T, 5] posteriors -> decoded class per frame [T]."""
    T = len(P)
    E = np.log(P[:, STATES] + EPS)                       # [T, n]
    n = len(STATES)
    delta = np.full(n, -np.inf)
    delta[FIRST] = E[0, FIRST]                           # start at the beginning of any state
    back = np.zeros((T, n), np.int32)
    for t in range(1, T):
        cand = delta[:, None] + LOGA                     # [from, to]
        back[t] = cand.argmax(0)
        delta = cand[back[t], np.arange(n)] + E[t]
    path = np.empty(T, np.int32)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return STATES[path]


def beats(cls, P):
    """Split the decoded sequence into beats and score them.

    Returns a list of dicts: start/end frame, duration, conf, flags and `usable`.
    """
    T = len(cls)
    # run-length encode
    edges = np.flatnonzero(np.diff(cls)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [T]])
    runs = list(zip(cls[starts], starts, ends))
    out = []
    for i in range(len(runs) - 4):
        seq = [runs[i + k][0] for k in range(5)]
        if seq != [0, 1, 2, 3, 0]:
            continue
        s, e = runs[i][1], runs[i + 4][1]
        if runs[i][1] == 0:                              # first S1 may be truncated
            continue
        conf = float(P[np.arange(s, e), cls[s:e]].mean())
        pnoise = float(P[s:e, 4].mean())
        out.append(dict(start=int(s), end=int(e), dur=(e - s) / FPS, conf=conf, pnoise=pnoise,
                        s1=(runs[i][2] - runs[i][1]) / FPS, s2=(runs[i + 2][2] - runs[i + 2][1]) / FPS))
    durs = np.array([b['dur'] for b in out])
    for j, b in enumerate(out):
        nb = durs[max(0, j - 3):j + 4]
        med = float(np.median(nb)) if len(nb) else b['dur']
        b['rhythm_dev'] = abs(b['dur'] - med) / med
        b['ok_dur'] = BEAT_MIN_S <= b['dur'] <= BEAT_MAX_S
        b['ok_conf'] = b['conf'] >= BEAT_CONF_MIN
        b['ok_rhythm'] = b['rhythm_dev'] <= RHYTHM_TOL or len(nb) < 3
        b['score'] = b['conf'] * (1.0 if b['ok_dur'] and b['ok_rhythm'] else 0.5)
        b['usable'] = bool(b['ok_dur'] and b['ok_conf'] and b['ok_rhythm'])
    return out


def recording_summary(bts, T):
    u = [b for b in bts if b['usable']]
    run = best = 0
    prev_end = None
    for b in bts:
        if b['usable'] and (prev_end is None or b['start'] == prev_end):
            run += 1
        elif b['usable']:
            run = 1
        else:
            run = 0
        prev_end = b['end']
        best = max(best, run)
    covered = sum(b['end'] - b['start'] for b in u)
    return dict(n_beats=len(bts), usable_beats=len(u), usable_frac_time=covered / max(T, 1),
                longest_usable_run=best,
                hr_bpm=60.0 / np.median([b['dur'] for b in u]) if u else 0.0)
