"""Rechnet die teuren Studien vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt `precomputed_sweep.json`.

  study  Ankünfte (24, 33, 36 Lkw/h) × Rückläufer (0, 20 %) × Streuung der Dauer (0, 0.25, 1, 4) an allen Stationen: je 4 Läufe à 300 000 Lkw; Gesamtzeit und mittlere Zahl je Station,
         dazu die Werte aus Jackson-Formel und Zerlegung
  block  Ankünfte (30, 36, 42 Lkw/h, ohne Rückläufer, exponentiell) × Platzgrenze des Stapels (4, 5, 6, 8, 12, unbegrenzt): je 4 Läufe à 150 000 Lkw"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jn_constants as C
import jn_evaluation as E
import jn_formulas as F
import jn_simulation as S


def _task(args):
    key, gamma_ph, r_pct, cs2, trucks, seed, cap = args
    gamma, p = E.model(gamma_ph, r_pct)
    caps = [None, None, cap]
    return key, S.simulate(gamma[0], C.SERVERS, C.MEANS, cs2, p, trucks, seed, cap=caps)


def main(workers):
    t0 = time.time()
    jobs, idx = [], 0
    for g in C.STUDY_GAMMA:
        for r in C.STUDY_R_PCT:
            for cs2 in C.STUDY_CS2:
                for rep in range(C.STUDY_REPS):
                    jobs.append((("study", g, r, cs2, None), g, r, cs2, C.STUDY_TRUCKS, 10_000 + 97 * idx + 1000 * rep, None))
                idx += 1
    for g in C.BLOCK_GAMMA:
        for cap in (*C.BLOCK_CAPS, None):
            for rep in range(C.BLOCK_REPS):
                jobs.append((("block", g, 0, 1.0, cap), g, 0, 1.0, C.BLOCK_TRUCKS, 500_000 + 97 * idx + 1000 * rep, cap))
            idx += 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_task, jobs, chunksize=1))
    groups = {}
    for key, res in out:
        groups.setdefault(key, []).append(res)
    study, block = [], []
    for (which, g, r, cs2, cap), runs in groups.items():
        summ = E.summarize_runs(runs)
        if which == "study":
            jack, qna = E.study_row(g, r, cs2)
            study.append({"gamma": g, "r_pct": r, "cs2": cs2, "jackson": jack, "qna": qna, **summ})
        else:
            gamma, p = E.model(g, 0)
            block.append({"gamma": g, "cap": cap, "jackson": F.jackson(gamma, p, C.SERVERS, C.MEANS)["total"], **summ})
    study.sort(key=lambda x: (x["gamma"], x["r_pct"], x["cs2"]))
    block.sort(key=lambda x: (x["gamma"], 10**9 if x["cap"] is None else x["cap"]))
    Path(E.PRECOMPUTED_PATH).write_text(json.dumps({"study_trucks": C.STUDY_TRUCKS, "study_reps": C.STUDY_REPS, "block_trucks": C.BLOCK_TRUCKS, "block_reps": C.BLOCK_REPS,
                                                    "study": study, "block": block}), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {E.PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(6, os.cpu_count() or 1))
