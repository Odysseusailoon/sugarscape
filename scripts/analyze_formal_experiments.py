#!/usr/bin/env python3
"""
Analyze the currently running (or completed) "formal" experiments.

Computes:
1) Survival & welfare: survival rate (current/latest), wealth gini (tick=0; later if checkpoints/metrics available)
2) Behavioral dynamics: trade success rate, deception rate (time series)
3) Moral alignment: external/self moral curves + alignment delta (final - initial)
4) Propagation specifics: conversion rate after contact with "good" agents + catalytic speedup (time to 80% coop)

Outputs (per run_dir):
  - plots/formal_metrics_timeseries.csv
  - plots/formal_summary.json
  - plots/formal_metrics.png

And a combined summary under:
  <formal_root>/_analysis/<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def gini(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    # G = (2 * sum(i*xi) / (n*sum(x))) - (n+1)/n
    cum = 0.0
    for i, x in enumerate(vals, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def latest_experiment_dir(exp_group_dir: Path) -> Optional[Path]:
    if not exp_group_dir.exists():
        return None
    candidates = [p for p in exp_group_dir.glob("experiment_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def read_csv_rows(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


@dataclass(frozen=True)
class RunPaths:
    name: str
    run_dir: Path
    debug_dir: Path
    initial_state: Path
    metrics_csv: Path
    trade_dialogues: Path
    reflections_jsonl: Path
    moral_scores: Path
    deaths: Path


def discover_formal_runs(formal_root: Path) -> List[RunPaths]:
    runs: List[RunPaths] = []
    for group in sorted([p for p in formal_root.iterdir() if p.is_dir()]):
        run_dir = latest_experiment_dir(group)
        if not run_dir:
            continue
        debug = run_dir / "debug"
        runs.append(
            RunPaths(
                name=group.name,
                run_dir=run_dir,
                debug_dir=debug,
                initial_state=run_dir / "initial_state.json",
                metrics_csv=run_dir / "metrics.csv",
                trade_dialogues=debug / "trade_dialogues.jsonl",
                reflections_jsonl=debug / "reflections.jsonl",
                moral_scores=debug / "moral_scores.csv",
                deaths=debug / "death_records.csv",
            )
        )
    return runs


def load_identities(initial_state_path: Path) -> Dict[int, str]:
    """agent_id -> origin_identity"""
    if not initial_state_path.exists():
        return {}
    s = load_json(initial_state_path)
    out: Dict[int, str] = {}
    for a in s.get("agents", []):
        aid = a.get("agent_id")
        ident = a.get("origin_identity")
        if isinstance(aid, int) and isinstance(ident, str):
            out[aid] = ident
    return out


def load_initial_wealth(initial_state_path: Path) -> Tuple[List[float], List[float]]:
    """Return (total_resources, sugar_only) at tick 0 for gini."""
    if not initial_state_path.exists():
        return ([], [])
    s = load_json(initial_state_path)
    totals: List[float] = []
    sugars: List[float] = []
    for a in s.get("agents", []):
        w = _safe_float(a.get("wealth"))
        sp = _safe_float(a.get("spice"))
        if w is None:
            continue
        sugars.append(w)
        totals.append(w + (sp or 0.0))
    return totals, sugars


def current_survival_rate(run: RunPaths) -> Dict[str, Any]:
    identities = load_identities(run.initial_state)
    initial_pop = len(identities) if identities else None
    deaths = 0
    if run.deaths.exists():
        deaths = sum(1 for _ in read_csv_rows(run.deaths))
    if initial_pop is None or initial_pop == 0:
        return {"initial_population": None, "deaths": deaths, "alive": None, "survival_rate": None}
    alive = max(0, initial_pop - deaths)
    return {"initial_population": initial_pop, "deaths": deaths, "alive": alive, "survival_rate": alive / initial_pop}


def trade_timeseries(run: RunPaths) -> Dict[int, Dict[str, Any]]:
    """
    Per tick:
      total_encounters = len(trade_dialogues records)
      completed
      deception_completed
      trade_success_rate = completed / total_encounters
      deception_rate = deception_completed / completed
    """
    by_tick: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "total_encounters": 0,
        "completed": 0,
        "deception_completed": 0,
    })
    for d in iter_jsonl(run.trade_dialogues):
        t = d.get("tick")
        if not isinstance(t, int):
            continue
        by_tick[t]["total_encounters"] += 1
        if d.get("outcome") == "completed":
            by_tick[t]["completed"] += 1
            if bool(d.get("deception_detected")):
                by_tick[t]["deception_completed"] += 1

    # derive rates
    out: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_tick):
        row = dict(by_tick[t])
        total = row["total_encounters"]
        comp = row["completed"]
        dece = row["deception_completed"]
        row["trade_success_rate"] = (comp / total) if total else None
        row["deception_rate"] = (dece / comp) if comp else None
        out[t] = row
    return out


def trade_repeat_reputation_timeseries(run: RunPaths) -> Dict[int, Dict[str, Any]]:
    """
    Per tick:
      - repeat_encounter_rate: fraction of encounters where the pair has met before (any prior tick)
      - mean_reputation_delta: average (rep_after - rep_before) across both agents (if present)
      - mean_trust_delta: average (trust_after - trust_before) across both directions (if present)
    """
    seen_pairs: set[tuple[int, int]] = set()
    by_tick: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "encounters": 0,
        "repeat_encounters": 0,
        "rep_delta_sum": 0.0,
        "rep_delta_n": 0,
        "trust_delta_sum": 0.0,
        "trust_delta_n": 0,
    })

    for d in iter_jsonl(run.trade_dialogues):
        t = d.get("tick")
        a = d.get("agent_a_id")
        b = d.get("agent_b_id")
        if not isinstance(t, int) or not isinstance(a, int) or not isinstance(b, int):
            continue

        pair = (a, b) if a < b else (b, a)
        by_tick[t]["encounters"] += 1
        if pair in seen_pairs:
            by_tick[t]["repeat_encounters"] += 1
        else:
            seen_pairs.add(pair)

        # reputation deltas (if present)
        ra0 = _safe_float(d.get("reputation_a_before"))
        ra1 = _safe_float(d.get("reputation_a_after"))
        rb0 = _safe_float(d.get("reputation_b_before"))
        rb1 = _safe_float(d.get("reputation_b_after"))
        if ra0 is not None and ra1 is not None:
            by_tick[t]["rep_delta_sum"] += (ra1 - ra0)
            by_tick[t]["rep_delta_n"] += 1
        if rb0 is not None and rb1 is not None:
            by_tick[t]["rep_delta_sum"] += (rb1 - rb0)
            by_tick[t]["rep_delta_n"] += 1

        # trust deltas (if present)
        ta0 = _safe_float(d.get("trust_a_to_b_before"))
        ta1 = _safe_float(d.get("trust_a_to_b_after"))
        tb0 = _safe_float(d.get("trust_b_to_a_before"))
        tb1 = _safe_float(d.get("trust_b_to_a_after"))
        if ta0 is not None and ta1 is not None:
            by_tick[t]["trust_delta_sum"] += (ta1 - ta0)
            by_tick[t]["trust_delta_n"] += 1
        if tb0 is not None and tb1 is not None:
            by_tick[t]["trust_delta_sum"] += (tb1 - tb0)
            by_tick[t]["trust_delta_n"] += 1

    out: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_tick):
        r = dict(by_tick[t])
        enc = r["encounters"]
        r["repeat_encounter_rate"] = (r["repeat_encounters"] / enc) if enc else None
        r["mean_reputation_delta"] = (r["rep_delta_sum"] / r["rep_delta_n"]) if r["rep_delta_n"] else None
        r["mean_trust_delta"] = (r["trust_delta_sum"] / r["trust_delta_n"]) if r["trust_delta_n"] else None
        # drop intermediate accumulators to keep outputs clean
        for k in ["rep_delta_sum", "rep_delta_n", "trust_delta_sum", "trust_delta_n"]:
            r.pop(k, None)
        out[t] = r
    return out


def moral_timeseries(run: RunPaths) -> Dict[int, Dict[str, Any]]:
    """Per tick mean of external/self and 6 dims from moral_scores.csv (event-based)."""
    by_tick_vals: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    if not run.moral_scores.exists():
        return {}
    for row in read_csv_rows(run.moral_scores):
        t = _safe_int(row.get("tick"))
        if t is None:
            continue
        for k in [
            "self_overall",
            "external_overall",
            "care",
            "fairness",
            "honesty",
            "respect",
            "exploitation",
            "harm",
        ]:
            v = _safe_float(row.get(k))
            if v is not None:
                by_tick_vals[t][k].append(v)

    out: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_tick_vals):
        vals = by_tick_vals[t]
        o: Dict[str, Any] = {"n_events": max((len(v) for v in vals.values()), default=0)}
        for k, xs in vals.items():
            o[f"{k}_mean"] = (sum(xs) / len(xs)) if xs else None
        out[t] = o
    return out


def reflection_abstraction_timeseries(run: RunPaths) -> Dict[int, Dict[str, Any]]:
    """
    Classify belief updates in reflections into:
      - concrete (partner-specific): keys starting with 'partner_' (e.g., partner_61.trustworthy)
      - abstract (world/norm/system): everything else in beliefs_changed

    Returns per tick:
      - abstraction_ratio = abstract / (abstract + concrete)
      - n_reflections, n_abstract, n_concrete
    """
    by_tick: Dict[int, Dict[str, int]] = defaultdict(lambda: {
        "n_reflections": 0,
        "n_abstract": 0,
        "n_concrete": 0,
    })
    for d in iter_jsonl(run.reflections_jsonl):
        t = d.get("tick")
        if not isinstance(t, int):
            continue
        by_tick[t]["n_reflections"] += 1
        changed = d.get("beliefs_changed") or []
        if not isinstance(changed, list):
            continue
        for k in changed:
            if not isinstance(k, str):
                continue
            if k.startswith("partner_"):
                by_tick[t]["n_concrete"] += 1
            else:
                by_tick[t]["n_abstract"] += 1

    out: Dict[int, Dict[str, Any]] = {}
    for t in sorted(by_tick):
        r = dict(by_tick[t])
        denom = r["n_abstract"] + r["n_concrete"]
        r["abstraction_ratio"] = (r["n_abstract"] / denom) if denom else None
        out[t] = r
    return out


def welfare_timeseries(run: RunPaths) -> Dict[int, Dict[str, Any]]:
    """Per tick macro welfare & inequality metrics from metrics.csv (logged periodically)."""
    if not run.metrics_csv.exists():
        return {}

    wanted = [
        # population & wealth
        "population",
        "mean_wealth",
        "gini",
        # welfare (social)
        "utilitarian_welfare",
        "average_welfare",
        "rawlsian_welfare",
        "nash_welfare",
        # welfare inequality & adjusted indices
        "welfare_gini",
        "gini_adjusted_welfare",
        "atkinson_index_05",
        "atkinson_adjusted_05",
        # survival
        "survival_rate",
    ]

    out: Dict[int, Dict[str, Any]] = {}
    for row in read_csv_rows(run.metrics_csv):
        t = _safe_int(row.get("tick"))
        if t is None:
            continue
        o: Dict[str, Any] = {}
        for k in wanted:
            if k == "population":
                o[k] = _safe_int(row.get(k))
            else:
                o[k] = _safe_float(row.get(k))
        out[t] = o
    return out


def alignment_delta(run: RunPaths) -> Dict[str, Any]:
    """
    Δ per agent = last external_overall - first external_overall.
    Uses moral_scores.csv (baseline_questionnaire at tick 0 is the initial if present).
    """
    if not run.moral_scores.exists():
        return {"n_agents": 0, "mean_delta": None, "by_identity": {}}

    first: Dict[int, float] = {}
    last: Dict[int, float] = {}
    last_tick: Dict[int, int] = {}
    first_tick: Dict[int, int] = {}
    agent_name: Dict[int, str] = {}
    for row in read_csv_rows(run.moral_scores):
        aid = _safe_int(row.get("agent_id"))
        t = _safe_int(row.get("tick"))
        ext = _safe_float(row.get("external_overall"))
        if aid is None or t is None or ext is None:
            continue
        agent_name[aid] = row.get("agent_name") or agent_name.get(aid, "")
        if aid not in first or t < first_tick.get(aid, 10**9):
            first[aid] = ext
            first_tick[aid] = t
        if aid not in last or t >= last_tick.get(aid, -1):
            last[aid] = ext
            last_tick[aid] = t

    deltas: Dict[int, float] = {}
    for aid in set(first) & set(last):
        deltas[aid] = last[aid] - first[aid]

    ids = load_identities(run.initial_state)
    by_ident: Dict[str, List[float]] = defaultdict(list)
    for aid, d in deltas.items():
        by_ident[ids.get(aid, "unknown")].append(d)

    def mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n_agents": len(deltas),
        "mean_delta": mean(list(deltas.values())),
        "by_identity": {k: {"n": len(v), "mean_delta": mean(v)} for k, v in sorted(by_ident.items())},
    }


def propagation_metrics(
    run: RunPaths,
    n_steps: int = 5,
) -> Dict[str, Any]:
    """
    Conversion Rate:
      Among targets with origin in {exploiter, survivor}, who had an encounter with a 'good' agent (altruist/pure_altruist),
      the fraction who later (within N ticks) completed at least one non-deceptive trade.
    Catalytic Speedup:
      Computes cooperative rate per tick = % agents that completed >=1 non-deceptive trade within a rolling window W.
      Reports first tick reaching 0.8, if any.
    """
    ids = load_identities(run.initial_state)
    if not ids:
        return {"exposed": 0, "converted": 0, "conversion_rate": None, "t80": None}

    good = {"altruist", "pure_altruist"}
    targets = {"exploiter", "survivor"}

    # completed non-deceptive trades per agent per tick
    completed_by_agent_tick: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    ticks_seen: set[int] = set()

    # exposures: agent -> list of exposure ticks
    exposure_ticks: Dict[int, List[int]] = defaultdict(list)

    for d in iter_jsonl(run.trade_dialogues):
        t = d.get("tick")
        if not isinstance(t, int):
            continue
        ticks_seen.add(t)
        a = d.get("agent_a_id")
        b = d.get("agent_b_id")
        if not isinstance(a, int) or not isinstance(b, int):
            continue
        ia = ids.get(a, "unknown")
        ib = ids.get(b, "unknown")

        # exposure logic (any encounter record)
        if ia in targets and ib in good:
            exposure_ticks[a].append(t)
        if ib in targets and ia in good:
            exposure_ticks[b].append(t)

        # conversion evidence
        if d.get("outcome") == "completed" and not bool(d.get("deception_detected")):
            completed_by_agent_tick[a][t] += 1
            completed_by_agent_tick[b][t] += 1

    if not ticks_seen:
        return {"exposed": 0, "converted": 0, "conversion_rate": None, "t80": None}

    max_tick = max(ticks_seen)

    exposed_agents = set(exposure_ticks.keys())
    converted = 0
    for aid in exposed_agents:
        # within N ticks after ANY exposure, did they complete a non-deceptive trade?
        exp_ts = exposure_ticks[aid]
        ok = False
        for et in exp_ts:
            for t in range(et + 1, min(max_tick, et + n_steps) + 1):
                if completed_by_agent_tick[aid].get(t, 0) > 0:
                    ok = True
                    break
            if ok:
                break
        if ok:
            converted += 1

    conversion_rate = (converted / len(exposed_agents)) if exposed_agents else None

    # catalytic speedup: coop rate by tick using rolling window
    W = 3
    coop_rate_by_tick: Dict[int, float] = {}
    agent_ids = list(ids.keys())
    for t in range(1, max_tick + 1):
        lo = max(1, t - W + 1)
        coop = 0
        for aid in agent_ids:
            has = False
            for tt in range(lo, t + 1):
                if completed_by_agent_tick[aid].get(tt, 0) > 0:
                    has = True
                    break
            if has:
                coop += 1
        coop_rate_by_tick[t] = coop / len(agent_ids) if agent_ids else 0.0

    t80 = None
    t90 = None
    for t, r in sorted(coop_rate_by_tick.items()):
        if t80 is None and r >= 0.8:
            t80 = t
        if t90 is None and r >= 0.9:
            t90 = t

    return {
        "exposed": len(exposed_agents),
        "converted": converted,
        "conversion_rate": conversion_rate,
        "t80": t80,
        "t90": t90,
        "coop_rate_by_tick": coop_rate_by_tick,
    }


def identity_shift_metrics(run: RunPaths) -> Dict[str, Any]:
    """
    Calculate Identity Shift (Δ self_identity_leaning).
    Table 3 metric: "Normie Identity Shift".
    """
    # 1. Initial leaning from initial_state.json
    if not run.initial_state.exists():
        return {}
    
    initial_leanings: Dict[int, float] = {}
    identities: Dict[int, str] = {}
    
    s = load_json(run.initial_state)
    for a in s.get("agents", []):
        aid = a.get("agent_id")
        if aid is not None:
            initial_leanings[aid] = float(a.get("self_identity_leaning", 0.0))
            identities[aid] = a.get("origin_identity", "unknown")

    # 2. Final leaning from reflections.jsonl (latest per agent)
    final_leanings: Dict[int, float] = {}
    if run.reflections_jsonl.exists():
        for row in iter_jsonl(run.reflections_jsonl):
            aid = _safe_int(row.get("agent_id")) # reflections usually have agent_id? 
            # Wait, reflections.jsonl structure in llm_agent.py doesn't explicitly save agent_id in the top level 
            # if it's just the history list dumped? 
            # Actually debug_logger.py logs reflections. Let's check if agent_id is there.
            # Assuming standard debug log format: {"tick": ..., "agent_id": ..., ...}
            if aid is None:
                continue
            
            # Check for identity_leaning_after
            leaning = _safe_float(row.get("identity_leaning_after"))
            if leaning is not None:
                final_leanings[aid] = leaning

    # If no reflections, final = initial (no shift)
    # But better to only count those who reflected? Or assume 0 shift?
    # Let's assume final = current state. If they never reflected, it's initial.
    
    deltas: Dict[int, float] = {}
    for aid, init_val in initial_leanings.items():
        final_val = final_leanings.get(aid, init_val)
        deltas[aid] = final_val - init_val

    # Group by identity
    by_ident: Dict[str, List[float]] = defaultdict(list)
    for aid, d in deltas.items():
        ident = identities.get(aid, "unknown")
        by_ident[ident].append(d)

    def mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    return {
        "mean_shift_all": mean(list(deltas.values())),
        "by_identity": {k: {"n": len(v), "mean_shift": mean(v)} for k, v in sorted(by_ident.items())}
    }


def cause_of_death_metrics(run: RunPaths) -> Dict[str, Any]:
    """
    Table A1: Cause of Death Analysis (The Filter).
    Groups: Survivors, Early Deaths (T<50), Late Deaths (T>150).
    Metrics: Avg Deception Rate, Avg Trade Attempts, Avg Partner Trust Score (Received).
    """
    # 1. Identify groups
    deaths: Dict[int, int] = {} # aid -> tick
    if run.deaths.exists():
        for row in read_csv_rows(run.deaths):
            aid = _safe_int(row.get("agent_id"))
            t = _safe_int(row.get("tick"))
            if aid is not None and t is not None:
                deaths[aid] = t

    ids = load_identities(run.initial_state)
    all_agents = set(ids.keys())
    survivors = all_agents - set(deaths.keys())
    
    early_deaths = {aid for aid, t in deaths.items() if t < 50}
    late_deaths = {aid for aid, t in deaths.items() if t > 150}
    
    groups = {
        "Survivors": survivors,
        "Early Deaths (<50)": early_deaths,
        "Late Deaths (>150)": late_deaths
    }

    # 2. Aggregate stats per agent
    # We need:
    # - Deception Rate (deceptions / completed trades)
    # - Trade Attempts (total encounters)
    # - Avg Partner Trust Score (Trust RECEIVED by agent)
    
    agent_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "attempts": 0,
        "completed": 0,
        "deceptions": 0,
        "trust_received_sum": 0.0,
        "trust_received_n": 0
    })

    for d in iter_jsonl(run.trade_dialogues):
        a = _safe_int(d.get("agent_a_id"))
        b = _safe_int(d.get("agent_b_id"))
        if a is None or b is None:
            continue
            
        # Attempts
        agent_stats[a]["attempts"] += 1
        agent_stats[b]["attempts"] += 1
        
        # Completed & Deception
        if d.get("outcome") == "completed":
            agent_stats[a]["completed"] += 1
            agent_stats[b]["completed"] += 1
            
            # Deception detected? 
            # Note: "deception_detected" usually means *someone* deceived.
            # We need to know WHO deceived.
            # The log usually has "deceiver_id" if we added it, or we infer from 
            # "deception_detected": true. 
            # Standard logs might not explicitly say WHO unless we parse `decider_id` for NO_TRADE,
            # but for COMPLETED trade with deception, it implies the one who made the offer/accept?
            # Actually, `trade.py` logs `deception_detected` if *actual* != *promised*.
            # It doesn't easily say *who*.
            # BUT, `DialogueTradeSystem` logs `deception_a` and `deception_b` booleans in recent versions?
            # Let's check `trade.py` or assume we can't perfectly attribute without `deception_a/b` fields.
            # If `deception_detected` is true, it's a "deceptive trade".
            # For "Avg Deception Rate" of an agent, we want (times THEY deceived / completed).
            # If the log doesn't split it, we might have to skip or use a proxy.
            # Let's look for `deception_a` / `deception_b` in recent logs.
            
            is_dec_a = bool(d.get("deception_a"))
            is_dec_b = bool(d.get("deception_b"))
            
            # Fallback: if only `deception_detected` is present, we don't know who.
            # But `trade.py` usually logs `actual_a` vs `promised_a`.
            # Let's try to infer if fields exist.
            if "deception_a" in d:
                if is_dec_a: agent_stats[a]["deceptions"] += 1
                if is_dec_b: agent_stats[b]["deceptions"] += 1
            elif d.get("deception_detected"):
                # Fallback: attribute to both? Or ignore? 
                # Attributing to both is noisy. Let's assume modern logs have it.
                pass

        # Trust Received (Trust B->A for A, Trust A->B for B)
        # trust_a_to_b_after -> Trust A holds for B. So B receives it.
        t_ab = _safe_float(d.get("trust_a_to_b_after"))
        t_ba = _safe_float(d.get("trust_b_to_a_after"))
        
        if t_ab is not None:
            agent_stats[b]["trust_received_sum"] += t_ab
            agent_stats[b]["trust_received_n"] += 1
        if t_ba is not None:
            agent_stats[a]["trust_received_sum"] += t_ba
            agent_stats[a]["trust_received_n"] += 1

    # 3. Compute group averages
    results = {}
    for gname, aids in groups.items():
        if not aids:
            results[gname] = None
            continue
            
        # Group metrics
        tot_attempts = 0
        tot_deceptions = 0
        tot_completed = 0
        tot_trust_sum = 0.0
        tot_trust_n = 0
        
        valid_agents = 0
        
        for aid in aids:
            s = agent_stats[aid]
            tot_attempts += s["attempts"]
            tot_deceptions += s["deceptions"]
            tot_completed += s["completed"]
            tot_trust_sum += s["trust_received_sum"]
            tot_trust_n += s["trust_received_n"]
            valid_agents += 1
            
        if valid_agents == 0:
            results[gname] = None
            continue

        avg_attempts = tot_attempts / valid_agents
        avg_deception_rate = (tot_deceptions / tot_completed) if tot_completed > 0 else 0.0
        avg_trust_received = (tot_trust_sum / tot_trust_n) if tot_trust_n > 0 else 0.5
        
        results[gname] = {
            "n": valid_agents,
            "avg_trade_attempts": avg_attempts,
            "avg_deception_rate": avg_deception_rate,
            "avg_partner_trust": avg_trust_received
        }
        
    return results


def seed_conversion_timeseries(
    run: RunPaths,
    conversion_window: int = 5,
) -> Dict[int, Dict[str, Any]]:
    """
    A cleaner "Seed -> Conversion" curve for the paper:
      - Exposure: target agent (exploiter/survivor) has ANY encounter with good agent (altruist/pure_altruist)
      - Conversion: within N ticks after FIRST exposure, target completes >=1 non-deceptive trade

    Per tick (cumulative):
      exposed_targets_cum, converted_targets_cum, conversion_rate_cum

    Notes:
      - Uses trade_dialogues.jsonl only (behavioral), so it's robust and cheap.
      - Intended as the "bad/normie being changed by good" visualization.
    """
    ids = load_identities(run.initial_state)
    if not ids:
        return {}

    good = {"altruist", "pure_altruist"}
    targets = {"exploiter", "survivor"}

    first_exposure: Dict[int, int] = {}
    first_conversion: Dict[int, int] = {}
    ticks_seen: set[int] = set()

    # Track completed non-deceptive by agent per tick
    completed_ok_by_agent_tick: Dict[int, set[int]] = defaultdict(set)

    # First pass: gather exposures + conversion evidence + tick range
    for d in iter_jsonl(run.trade_dialogues):
        t = d.get("tick")
        a = d.get("agent_a_id")
        b = d.get("agent_b_id")
        if not isinstance(t, int) or not isinstance(a, int) or not isinstance(b, int):
            continue
        ticks_seen.add(t)

        ia = ids.get(a, "unknown")
        ib = ids.get(b, "unknown")

        # exposures
        if ia in targets and ib in good:
            first_exposure.setdefault(a, t)
        if ib in targets and ia in good:
            first_exposure.setdefault(b, t)

        # conversion evidence (non-deceptive completed trades)
        if d.get("outcome") == "completed" and not bool(d.get("deception_detected")):
            completed_ok_by_agent_tick[a].add(t)
            completed_ok_by_agent_tick[b].add(t)

    if not ticks_seen:
        return {}

    max_tick = max(ticks_seen)

    # Determine first conversion tick per exposed agent (within window)
    for aid, et in first_exposure.items():
        for t in range(et + 1, min(max_tick, et + conversion_window) + 1):
            if t in completed_ok_by_agent_tick.get(aid, set()):
                first_conversion[aid] = t
                break

    # Build cumulative curve
    by_tick: Dict[int, Dict[str, Any]] = {}
    for t in range(1, max_tick + 1):
        exposed = [aid for aid, et in first_exposure.items() if et <= t]
        converted = [aid for aid, ct in first_conversion.items() if ct <= t]
        exp_n = len(exposed)
        conv_n = len(converted)
        by_tick[t] = {
            "exposed_targets_cum": exp_n,
            "converted_targets_cum": conv_n,
            "conversion_rate_cum": (conv_n / exp_n) if exp_n else None,
        }

    # Write per-run CSV + plot
    out_dir = run.run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "seed_conversion_curve.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tick", "exposed_targets_cum", "converted_targets_cum", "conversion_rate_cum"])
        w.writeheader()
        for t in sorted(by_tick):
            w.writerow({"tick": t, **by_tick[t]})

    # Plot (if matplotlib)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        xs = list(sorted(by_tick))
        ys = [by_tick[t]["conversion_rate_cum"] for t in xs]
        fig = plt.figure(figsize=(6.8, 4.6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys, marker="o", linewidth=2)
        ax.set_title("Seed → Conversion (cumulative)")
        ax.set_xlabel("tick")
        ax.set_ylabel("conversion_rate_cum")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "seed_conversion_curve.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    return by_tick

def write_outputs(
    run: RunPaths,
    summary: Dict[str, Any],
    timeseries_rows: List[Dict[str, Any]],
    make_plots: bool = True,
) -> None:
    out_dir = run.run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV time series
    ts_path = out_dir / "formal_metrics_timeseries.csv"
    if timeseries_rows:
        with ts_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(timeseries_rows[0].keys()))
            w.writeheader()
            for r in timeseries_rows:
                w.writerow(r)

    # JSON summary
    with (out_dir / "formal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not make_plots:
        return

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not timeseries_rows:
        return

    def _nan(x: Any) -> float:
        try:
            if x is None or x == "":
                return float("nan")
            return float(x)
        except Exception:
            return float("nan")

    ticks = [r["tick"] for r in timeseries_rows]
    ext = [_nan(r.get("external_overall_mean")) for r in timeseries_rows]
    selfv = [_nan(r.get("self_overall_mean")) for r in timeseries_rows]
    succ = [_nan(r.get("trade_success_rate")) for r in timeseries_rows]
    dece = [_nan(r.get("deception_rate")) for r in timeseries_rows]
    coop = [_nan(r.get("cooperative_rate")) for r in timeseries_rows]

    util = [_nan(r.get("utilitarian_welfare")) for r in timeseries_rows]
    rawls = [_nan(r.get("rawlsian_welfare")) for r in timeseries_rows]
    nash = [_nan(r.get("nash_welfare")) for r in timeseries_rows]
    gini_wealth = [_nan(r.get("gini")) for r in timeseries_rows]
    gini_welfare = [_nan(r.get("welfare_gini")) for r in timeseries_rows]
    mean_wealth = [_nan(r.get("mean_wealth")) for r in timeseries_rows]
    igi_pop = [_nan(r.get("igi_population_mean")) for r in timeseries_rows]
    abstr = [_nan(r.get("abstraction_ratio")) for r in timeseries_rows]

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 14))

    ax = axes[0, 0]
    ax.plot(ticks, ext, "o-", label="external_overall_mean")
    ax.plot(ticks, selfv, "o-", label="self_overall_mean")
    ax.set_title("Moral alignment (mean)")
    ax.set_xlabel("tick")
    ax.set_ylabel("score")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ticks, succ, "o-", label="trade_success_rate")
    ax.set_title("Trade success rate")
    ax.set_xlabel("tick")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(ticks, dece, "o-", label="deception_rate (among completed)")
    ax.set_title("Deception rate")
    ax.set_xlabel("tick")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(ticks, coop, "o-", label="cooperative_rate (rolling window)")
    ax.axhline(0.8, linestyle="--", alpha=0.6, color="gray")
    ax.set_title("Cooperative rate")
    ax.set_xlabel("tick")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(ticks, util, "o-", label="utilitarian_welfare")
    ax.plot(ticks, rawls, "o-", label="rawlsian_welfare")
    ax.plot(ticks, nash, "o-", label="nash_welfare")
    ax.set_title("Social welfare")
    ax.set_xlabel("tick")
    ax.set_ylabel("welfare")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(ticks, gini_wealth, "o-", label="wealth_gini")
    ax.plot(ticks, gini_welfare, "o-", label="welfare_gini")
    ax2 = ax.twinx()
    ax2.plot(ticks, mean_wealth, "o--", alpha=0.7, label="mean_wealth")
    ax.set_title("Inequality + mean wealth")
    ax.set_xlabel("tick")
    ax.set_ylabel("gini")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    # combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    ax = axes[3, 0]
    ax.plot(ticks, igi_pop, "o-", label="IGI population mean")
    ax.set_title("Infinite Game Index (population mean)")
    ax.set_xlabel("tick")
    ax.set_ylabel("index")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 1]
    ax.plot(ticks, abstr, "o-", label="abstraction_ratio")
    ax.set_title("Belief abstraction ratio (world/norm vs partner)")
    ax.set_xlabel("tick")
    ax.set_ylabel("ratio")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "formal_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ols_line(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float]]:
    """Return (slope, intercept) for y = slope*x + intercept."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept


def igi_timeseries_and_agents(
    run: RunPaths,
    max_tick_hint: Optional[int] = None,
    window: int = 5,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Implements the paper-friendly Infinite Game Index (IGI), per your spec:
      IGI_{i,t} = (1 - D_{i,t}) * (1 + R_{i,t}) * T_{i,t}

    D_{i,t}: moving average deception rate for agent i (window W) over completed trades
    R_{i,t}: moving average retention rate (window W): fraction of completed trades with previously traded partners
    T_{i,t}: current reputation score (0..1), as tracked by the environment (from trade_dialogues fields)

    Returns:
      - per_tick: tick -> {igi_population_mean, igi_n_agents, igi_D_mean, igi_R_mean, igi_T_mean}
      - per_agent: rows with avg_igi, lifespan, died, identity
    """
    # track current reputation (default neutral)
    rep: Dict[int, float] = defaultdict(lambda: 0.5)
    # partners ever traded with (completed)
    partners: Dict[int, set[int]] = defaultdict(set)
    # rolling windows (as lists capped to W)
    dece_hist: Dict[int, List[int]] = defaultdict(list)
    ret_hist: Dict[int, List[int]] = defaultdict(list)

    # per-agent IGI accumulation (for lifetime avg)
    igi_sum: Dict[int, float] = defaultdict(float)
    igi_n: Dict[int, int] = defaultdict(int)

    # record last processed tick per agent to snapshot IGI
    by_tick_agents: Dict[int, set[int]] = defaultdict(set)

    max_tick = 0
    for d in iter_jsonl(run.trade_dialogues):
        t = d.get("tick")
        a = d.get("agent_a_id")
        b = d.get("agent_b_id")
        if not isinstance(t, int) or not isinstance(a, int) or not isinstance(b, int):
            continue
        max_tick = max(max_tick, t)

        # update reputation if present (use *_after as current state at tick t)
        ra1 = _safe_float(d.get("reputation_a_after"))
        rb1 = _safe_float(d.get("reputation_b_after"))
        if ra1 is not None:
            rep[a] = ra1
        if rb1 is not None:
            rep[b] = rb1

        # only completed trades contribute to deception/retention
        if d.get("outcome") != "completed":
            continue

        dece = 1 if bool(d.get("deception_detected")) else 0

        # retention: did they trade with an old partner?
        a_ret = 1 if b in partners[a] else 0
        b_ret = 1 if a in partners[b] else 0

        def push(hist: List[int], v: int) -> None:
            hist.append(v)
            if len(hist) > window:
                del hist[0]

        push(dece_hist[a], dece)
        push(dece_hist[b], dece)
        push(ret_hist[a], a_ret)
        push(ret_hist[b], b_ret)

        partners[a].add(b)
        partners[b].add(a)

        by_tick_agents[t].add(a)
        by_tick_agents[t].add(b)

    if max_tick_hint is not None:
        max_tick = max(max_tick, max_tick_hint)

    # compute IGI per agent per tick AFTER processing that tick's encounters
    per_tick: Dict[int, Dict[str, Any]] = {}
    for t in range(1, max_tick + 1):
        # agents that have any completed-trade history up to this tick
        active_agents = {aid for tt in range(1, t + 1) for aid in by_tick_agents.get(tt, set())}
        if not active_agents:
            continue

        igis: List[float] = []
        Ds: List[float] = []
        Rs: List[float] = []
        Ts: List[float] = []

        for aid in active_agents:
            dh = dece_hist.get(aid) or []
            rh = ret_hist.get(aid) or []
            if not dh or not rh:
                continue

            D = sum(dh) / len(dh) if dh else 0.0
            R = sum(rh) / len(rh) if rh else 0.0
            T = rep.get(aid, 0.5)
            # clamp to [0,1] where meaningful
            D = max(0.0, min(1.0, D))
            R = max(0.0, min(1.0, R))
            T = max(0.0, min(1.0, T))

            igi = (1.0 - D) * (1.0 + R) * T
            igis.append(igi)
            Ds.append(D)
            Rs.append(R)
            Ts.append(T)

            igi_sum[aid] += igi
            igi_n[aid] += 1

        if igis:
            per_tick[t] = {
                "igi_population_mean": sum(igis) / len(igis),
                "igi_n_agents": len(igis),
                "igi_D_mean": sum(Ds) / len(Ds) if Ds else None,
                "igi_R_mean": sum(Rs) / len(Rs) if Rs else None,
                "igi_T_mean": sum(Ts) / len(Ts) if Ts else None,
            }

    # lifespan per agent
    death_tick: Dict[int, int] = {}
    if run.deaths.exists():
        for row in read_csv_rows(run.deaths):
            aid = _safe_int(row.get("agent_id"))
            t = _safe_int(row.get("tick"))
            if aid is None or t is None:
                continue
            death_tick[aid] = t

    ids = load_identities(run.initial_state)
    population_ids = sorted(ids.keys())
    if max_tick_hint is None:
        max_tick_hint = max_tick

    agent_rows: List[Dict[str, Any]] = []
    for aid in population_ids:
        died = aid in death_tick
        lifespan = death_tick.get(aid, max_tick_hint or 0)
        avg_igi = (igi_sum[aid] / igi_n[aid]) if igi_n.get(aid, 0) else None
        agent_rows.append(
            {
                "agent_id": aid,
                "origin_identity": ids.get(aid, "unknown"),
                "died": died,
                "lifespan_ticks": lifespan,
                "avg_igi": avg_igi,
                "final_reputation": rep.get(aid, 0.5),
                "igi_n_points": igi_n.get(aid, 0),
            }
        )

    # write per-run artifacts
    out_dir = run.run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "igi_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        if per_tick:
            w = csv.DictWriter(f, fieldnames=["tick"] + list(next(iter(per_tick.values())).keys()))
            w.writeheader()
            for t in sorted(per_tick):
                w.writerow({"tick": t, **per_tick[t]})

    with (out_dir / "igi_agent_summary.csv").open("w", newline="", encoding="utf-8") as f:
        if agent_rows:
            w = csv.DictWriter(f, fieldnames=list(agent_rows[0].keys()))
            w.writeheader()
            for r in agent_rows:
                w.writerow(r)

    # scatter plot (avg_igi vs lifespan)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        xs: List[float] = []
        ys: List[float] = []
        for r in agent_rows:
            if r["avg_igi"] is None:
                continue
            xs.append(float(r["avg_igi"]))
            ys.append(float(r["lifespan_ticks"]))
        if len(xs) >= 3:
            fig = plt.figure(figsize=(6.5, 4.8))
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(xs, ys, s=18, alpha=0.6)
            line = _ols_line(xs, ys)
            if line is not None:
                m, b = line
                x0, x1 = min(xs), max(xs)
                ax.plot([x0, x1], [m * x0 + b, m * x1 + b], linewidth=2, alpha=0.9)
            ax.set_title("Survival correlation: avg IGI vs lifespan")
            ax.set_xlabel("avg IGI (lifetime)")
            ax.set_ylabel("lifespan (ticks)")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "igi_survival_scatter.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    return per_tick, agent_rows



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-root", type=str, default="results/sugarscape/formal")
    ap.add_argument("--conversion-window", type=int, default=5, help="N ticks after exposure to count conversion")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    formal_root = Path(args.formal_root).resolve()
    runs = discover_formal_runs(formal_root)
    if not runs:
        raise SystemExit(f"No runs found under {formal_root}")

    # combined analysis dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_dir = formal_root / "_analysis" / ts
    combined_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = combined_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, Any]] = []
    combined_timeseries_long: List[Dict[str, Any]] = []
    by_run_timeseries: Dict[str, List[Dict[str, Any]]] = {}
    by_run_seed_curve: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for run in runs:
        surv = current_survival_rate(run)
        totals, sugars = load_initial_wealth(run.initial_state)
        gini_total_0 = gini(totals)
        gini_sugar_0 = gini(sugars)

        trades = trade_timeseries(run)
        repeats = trade_repeat_reputation_timeseries(run)
        morals = moral_timeseries(run)
        welfare = welfare_timeseries(run)
        abstr = reflection_abstraction_timeseries(run)
        align = alignment_delta(run)
        id_shift = identity_shift_metrics(run)
        death_metrics = cause_of_death_metrics(run)
        prop = propagation_metrics(run, n_steps=args.conversion_window)
        seed_curve = seed_conversion_timeseries(run, conversion_window=args.conversion_window)
        by_run_seed_curve[run.name] = seed_curve

        # IGI (per-agent behavioral mode) + survival correlation scatter
        max_tick_hint = max(
            [0]
            + list(trades.keys())
            + list(morals.keys())
            + list(welfare.keys())
            + list(abstr.keys())
            + list(prop.get("coop_rate_by_tick", {}).keys())
        )
        igi_by_tick, _igi_agents = igi_timeseries_and_agents(run, max_tick_hint=max_tick_hint, window=5)

        # build per-tick merged rows (union ticks)
        all_ticks = sorted(
            set(trades.keys())
            | set(repeats.keys())
            | set(morals.keys())
            | set(welfare.keys())
            | set(abstr.keys())
            | set(igi_by_tick.keys())
            | set(prop.get("coop_rate_by_tick", {}).keys())
        )
        ts_rows: List[Dict[str, Any]] = []
        for t in all_ticks:
            tr = trades.get(t, {})
            rp = repeats.get(t, {})
            mr = morals.get(t, {})
            wf = welfare.get(t, {})
            ab = abstr.get(t, {})
            coop = prop.get("coop_rate_by_tick", {}).get(t)
            ig = igi_by_tick.get(t, {})

            ts_rows.append(
                {
                    "tick": t,
                    "total_encounters": tr.get("total_encounters"),
                    "completed": tr.get("completed"),
                    "trade_success_rate": tr.get("trade_success_rate"),
                    "deception_rate": tr.get("deception_rate"),
                    "repeat_encounter_rate": rp.get("repeat_encounter_rate"),
                    "mean_reputation_delta": rp.get("mean_reputation_delta"),
                    "mean_trust_delta": rp.get("mean_trust_delta"),
                    "external_overall_mean": mr.get("external_overall_mean"),
                    "self_overall_mean": mr.get("self_overall_mean"),
                    "cooperative_rate": coop,
                    # macro welfare + inequality
                    "population": wf.get("population"),
                    "mean_wealth": wf.get("mean_wealth"),
                    "gini": wf.get("gini"),
                    "utilitarian_welfare": wf.get("utilitarian_welfare"),
                    "average_welfare": wf.get("average_welfare"),
                    "rawlsian_welfare": wf.get("rawlsian_welfare"),
                    "nash_welfare": wf.get("nash_welfare"),
                    "welfare_gini": wf.get("welfare_gini"),
                    "gini_adjusted_welfare": wf.get("gini_adjusted_welfare"),
                    "atkinson_index_05": wf.get("atkinson_index_05"),
                    "atkinson_adjusted_05": wf.get("atkinson_adjusted_05"),
                    "survival_rate_macro": wf.get("survival_rate"),
                    # abstraction / infinite-game proxies
                    "abstraction_ratio": ab.get("abstraction_ratio"),
                    "n_reflections": ab.get("n_reflections"),
                    "igi_population_mean": ig.get("igi_population_mean"),
                    "igi_n_agents": ig.get("igi_n_agents"),
                    "igi_D_mean": ig.get("igi_D_mean"),
                    "igi_R_mean": ig.get("igi_R_mean"),
                    "igi_T_mean": ig.get("igi_T_mean"),
                }
            )

        latest_wf = welfare.get(max(welfare.keys()), {}) if welfare else {}
        latest_ab = abstr.get(max(abstr.keys()), {}) if abstr else {}
        summary = {
            "run_name": run.name,
            "run_dir": str(run.run_dir),
            "survival": surv,
            "gini_tick0_total_resources": gini_total_0,
            "gini_tick0_sugar_only": gini_sugar_0,
            "latest_macro_metrics": latest_wf,
            "latest_abstraction_metrics": latest_ab,
            "alignment_delta": align,
            "identity_shift": id_shift,
            "cause_of_death_analysis": death_metrics,
            "propagation": {k: v for k, v in prop.items() if k != "coop_rate_by_tick"},
        }

        write_outputs(run, summary, ts_rows, make_plots=not args.no_plots)
        by_run_timeseries[run.name] = ts_rows
        for r in ts_rows:
            combined_timeseries_long.append({"run_name": run.name, **r})

        # Copy per-run key figures into one place
        if not args.no_plots:
            copies = [
                (run.run_dir / "plots" / "formal_metrics.png", f"{run.name}__formal_metrics.png"),
                (run.run_dir / "plots" / "igi_survival_scatter.png", f"{run.name}__igi_survival_scatter.png"),
                (run.run_dir / "plots" / "seed_conversion_curve.png", f"{run.name}__seed_conversion_curve.png"),
                (run.run_dir / "plots" / "igi_timeseries.csv", f"{run.name}__igi_timeseries.csv"),
                (run.run_dir / "plots" / "igi_agent_summary.csv", f"{run.name}__igi_agent_summary.csv"),
                (run.run_dir / "plots" / "seed_conversion_curve.csv", f"{run.name}__seed_conversion_curve.csv"),
            ]
            for src, dst_name in copies:
                try:
                    if src.exists():
                        shutil.copy2(src, figures_dir / dst_name)
                except Exception:
                    pass

        # combined one-row snapshot (latest tick)
        latest_tick = max(all_ticks) if all_ticks else None
        latest_ext = None
        latest_succ = None
        latest_dece = None
        latest_coop = None
        latest_macro = None
        if latest_tick is not None:
            latest_ext = morals.get(latest_tick, {}).get("external_overall_mean")
            latest_succ = trades.get(latest_tick, {}).get("trade_success_rate")
            latest_dece = trades.get(latest_tick, {}).get("deception_rate")
            latest_coop = prop.get("coop_rate_by_tick", {}).get(latest_tick)
            latest_macro = welfare.get(latest_tick)

        combined_rows.append(
            {
                "run_name": run.name,
                "run_dir": str(run.run_dir),
                "latest_tick": latest_tick,
                "alive": surv.get("alive"),
                "survival_rate": surv.get("survival_rate"),
                "gini_tick0_total_resources": gini_total_0,
                # macro welfare / inequality at latest logged tick (may be sparse)
                "mean_wealth_latest": (latest_macro or {}).get("mean_wealth"),
                "gini_latest": (latest_macro or {}).get("gini"),
                "utilitarian_welfare_latest": (latest_macro or {}).get("utilitarian_welfare"),
                "average_welfare_latest": (latest_macro or {}).get("average_welfare"),
                "rawlsian_welfare_latest": (latest_macro or {}).get("rawlsian_welfare"),
                "nash_welfare_latest": (latest_macro or {}).get("nash_welfare"),
                "welfare_gini_latest": (latest_macro or {}).get("welfare_gini"),
                "atkinson_index_05_latest": (latest_macro or {}).get("atkinson_index_05"),
                "trade_success_rate_latest": latest_succ,
                "deception_rate_latest": latest_dece,
                "external_overall_mean_latest": latest_ext,
                "alignment_delta_mean": align.get("mean_delta"),
                "identity_shift_mean": id_shift.get("mean_shift_all"),
                "conversion_rate": prop.get("conversion_rate"),
                "t80": prop.get("t80"),
                "t90": prop.get("t90"),
                "cooperative_rate_latest": latest_coop,
            }
        )

    # write combined CSV/JSON
    combined_csv = combined_dir / "formal_combined_snapshot.csv"
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(combined_rows[0].keys()))
        w.writeheader()
        for r in combined_rows:
            w.writerow(r)

    with (combined_dir / "formal_combined_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(combined_rows, f, indent=2)

    # write combined long-format time series
    combined_ts_csv = combined_dir / "formal_combined_timeseries.csv"
    if combined_timeseries_long:
        with combined_ts_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(combined_timeseries_long[0].keys()))
            w.writeheader()
            for r in combined_timeseries_long:
                w.writerow(r)

    # comparison plot across runs
    if not args.no_plots and by_run_timeseries:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None:
            def _nan(x: Any) -> float:
                try:
                    if x is None or x == "":
                        return float("nan")
                    return float(x)
                except Exception:
                    return float("nan")

            # union ticks across runs
            all_ticks: List[int] = sorted(
                {int(r["tick"]) for rows in by_run_timeseries.values() for r in rows if r.get("tick") is not None}
            )

            def series(rows: List[Dict[str, Any]], key: str) -> List[float]:
                by_t = {int(r["tick"]): r.get(key) for r in rows if r.get("tick") is not None}
                return [_nan(by_t.get(t)) for t in all_ticks]

            fig, axes = plt.subplots(4, 2, figsize=(14.5, 14))

            # (0,0) moral external mean
            ax = axes[0, 0]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "external_overall_mean"), marker="o", linewidth=1.5, label=name)
            ax.set_title("External moral (mean)")
            ax.set_xlabel("tick")
            ax.set_ylabel("score")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # (0,1) trade success
            ax = axes[0, 1]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "trade_success_rate"), marker="o", linewidth=1.5, label=name)
            ax.set_title("Trade success rate")
            ax.set_xlabel("tick")
            ax.set_ylabel("rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # (1,0) deception rate
            ax = axes[1, 0]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "deception_rate"), marker="o", linewidth=1.5, label=name)
            ax.set_title("Deception rate (among completed)")
            ax.set_xlabel("tick")
            ax.set_ylabel("rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # (1,1) social welfare
            ax = axes[1, 1]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "utilitarian_welfare"), marker="o", linewidth=1.5, label=f"{name} util")
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "rawlsian_welfare"), marker=".", linewidth=1.2, alpha=0.7, label=f"{name} rawls")
            ax.set_title("Social welfare (utilitarian & rawlsian)")
            ax.set_xlabel("tick")
            ax.set_ylabel("welfare")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

            # (2,0) inequality (wealth gini + welfare gini)
            ax = axes[2, 0]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "gini"), marker="o", linewidth=1.5, label=f"{name} wealth_gini")
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "welfare_gini"), marker=".", linewidth=1.2, alpha=0.7, label=f"{name} welfare_gini")
            ax.set_title("Inequality (wealth gini + welfare gini)")
            ax.set_xlabel("tick")
            ax.set_ylabel("gini")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

            # (2,1) mean wealth
            ax = axes[2, 1]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "mean_wealth"), marker="o", linewidth=1.5, label=name)
            ax.set_title("Mean wealth")
            ax.set_xlabel("tick")
            ax.set_ylabel("mean_wealth")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # (3,0) IGI (population mean)
            ax = axes[3, 0]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "igi_population_mean"), marker="o", linewidth=1.5, label=name)
            ax.set_title("Infinite Game Index (population mean)")
            ax.set_xlabel("tick")
            ax.set_ylabel("index")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # (3,1) abstraction ratio
            ax = axes[3, 1]
            for name, rows in by_run_timeseries.items():
                ax.plot(all_ticks, series(rows, "abstraction_ratio"), marker="o", linewidth=1.5, label=name)
            ax.set_title("Belief abstraction ratio")
            ax.set_xlabel("tick")
            ax.set_ylabel("ratio")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(combined_dir / "formal_compare.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

            # Killer Visual A: Hell vs Heaven trajectory (if present)
            hell_key = None
            heaven_key = None
            for k in by_run_timeseries.keys():
                if "exploiter_hybrid" in k and "v4_eval" in k:
                    hell_key = k
                if "exploiter_abundant_nopressure" in k and "v4_eval" in k:
                    heaven_key = k
            if hell_key and heaven_key:
                fig = plt.figure(figsize=(7.2, 4.6))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(all_ticks, series(by_run_timeseries[hell_key], "igi_population_mean"), marker="o", label=f"Hell: {hell_key}")
                ax.plot(all_ticks, series(by_run_timeseries[heaven_key], "igi_population_mean"), marker="o", label=f"Heaven: {heaven_key}")
                ax.set_title("Hell vs Heaven: Population Avg IGI")
                ax.set_xlabel("tick")
                ax.set_ylabel("IGI")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(combined_dir / "hell_vs_heaven_igi.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

            # Seed effect compare: conversion_rate_cum over time (targets exposed to good agents)
            if by_run_seed_curve:
                fig = plt.figure(figsize=(7.6, 5.0))
                ax = fig.add_subplot(1, 1, 1)
                for name, curve in sorted(by_run_seed_curve.items()):
                    if not curve:
                        continue
                    xs = sorted(curve.keys())
                    ys = [curve[t].get("conversion_rate_cum") for t in xs]
                    # only plot if there is any exposure
                    if all(curve[t].get("exposed_targets_cum", 0) == 0 for t in xs):
                        continue
                    ax.plot(xs, ys, marker="o", linewidth=2, label=name)
                ax.set_title("Seed Effect: conversion after exposure (cumulative)")
                ax.set_xlabel("tick")
                ax.set_ylabel("conversion_rate_cum")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(combined_dir / "seed_conversion_compare.png", dpi=180, bbox_inches="tight")
                plt.close(fig)

    print(f"Wrote combined snapshot: {combined_csv}")


if __name__ == "__main__":
    main()

