#!/usr/bin/env python3
"""Conformance of a `/v1/systemone` server to the hosted System One forms: 20 requests
(`cases.json`), each checked for status and answer *shape* (types and keys, not values),
plus the three routes around it. A case whose `expect` is "4xx|200" passes either way and
reports which (the option limit differs per readout).

    python3 check.py http://127.0.0.1:8090 [--model minicpm5-2b] [--timeout 60] [-v]

Exit 0 when every case passes. Extra keys in a response are reported, never failed.
Standard library only.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOL = 0.011  # probabilities are printed to four decimals; sums may miss 1 by rounding


def call(method, url, body=None, timeout=60, headers=None):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status, hdrs = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        raw, status, hdrs = e.read(), e.code, dict(e.headers)
    ms = (time.perf_counter() - t) * 1000
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = raw.decode(errors="replace")
    return status, parsed, ms, hdrs


def substitute(obj, model):
    if isinstance(obj, dict):
        return {k: substitute(v, model) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, model) for v in obj]
    return model if obj == "$MODEL" else obj


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_answer(qid, question, answer, problems, extras):
    if not isinstance(answer, dict):
        problems.append(f"{qid}: answer is not an object")
        return
    qtype = question["type"]
    if answer.get("type") != qtype:
        problems.append(f"{qid}: type {answer.get('type')!r} != {qtype!r}")
    if qtype == "noul":
        p = answer.get("noul")
        if not (is_num(p) and 0 <= p <= 1):
            problems.append(f"{qid}: 'noul' must be a number in [0, 1], got {p!r}")
        extras.update(f"{qtype}.{k}" for k in answer if k not in ("type", "noul"))
        return
    conf = answer.get("confidence")
    if not (is_num(conf) and 0 <= conf <= 1):
        problems.append(f"{qid}: 'confidence' must be a number in [0, 1], got {conf!r}")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not all(is_num(v) and 0 <= v <= 1 for v in probs.values()):
        problems.append(f"{qid}: 'probabilities' must map keys to numbers in [0, 1]")
        return
    if abs(sum(probs.values()) - 1) > TOL:
        problems.append(f"{qid}: probabilities sum to {sum(probs.values()):.4f}")
    criteria = question.get("criteria")
    if qtype == "choice":
        keys = list(criteria.keys()) if isinstance(criteria, dict) else [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in criteria]
        if set(probs) != set(keys):
            problems.append(f"{qid}: probability keys {sorted(probs)} != options {sorted(keys)}")
        choice = answer.get("choice")
        if choice not in probs:
            problems.append(f"{qid}: 'choice' {choice!r} is not one of the options")
        elif probs and probs[choice] < max(probs.values()) - 1e-9:
            problems.append(f"{qid}: 'choice' {choice!r} is not the highest-probability option")
        extras.update(f"{qtype}.{k}" for k in answer if k not in ("type", "choice", "probabilities", "confidence"))
    elif qtype == "score":
        n = len(criteria)
        levels = [str(i) for i in range(n)]
        if set(probs) != set(levels):
            problems.append(f"{qid}: probability keys {sorted(probs)} != levels {levels}")
        legend = answer.get("legend")
        if not isinstance(legend, dict) or set(legend) != set(levels) or not all(isinstance(v, str) for v in legend.values()):
            problems.append(f"{qid}: 'legend' must map every level '0'..'{n-1}' to a string")
        score = answer.get("score")
        if not (is_num(score) and -TOL <= score <= n - 1 + TOL):
            problems.append(f"{qid}: 'score' must be a number in [0, {n-1}], got {score!r}")
        elif set(probs) == set(levels):
            expected = sum(int(k) * v for k, v in probs.items())
            if abs(expected - score) > 0.02:
                problems.append(f"{qid}: 'score' {score} != probability-weighted level {expected:.4f}")
        extras.update(f"{qtype}.{k}" for k in answer if k not in ("type", "score", "legend", "probabilities", "confidence"))


def check_response(request, body, problems, extras):
    if not isinstance(body, dict):
        problems.append("response is not a JSON object")
        return
    for key in ("model", "answers", "usage"):
        if key not in body:
            problems.append(f"missing top-level '{key}'")
    if not isinstance(body.get("model"), str):
        problems.append("'model' must be a string")
    usage = body.get("usage")
    if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) for k in ("input_tokens", "output_tokens")):
        problems.append("'usage' must have integer 'input_tokens' and 'output_tokens'")
    answers = body.get("answers")
    if not isinstance(answers, dict):
        problems.append("'answers' must be an object")
        return
    if set(answers) != set(request["questions"]):
        problems.append(f"answer ids {sorted(answers)} != question ids {sorted(request['questions'])}")
    for qid, q in request["questions"].items():
        if qid in answers:
            check_answer(qid, q, answers[qid], problems, extras)
    extras.update(f"top.{k}" for k in body if k not in ("model", "answers", "usage"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_url")
    ap.add_argument("--model", help="model id to send; default: the first model the server lists")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--api-key", help="sent as a bearer token when the server wants one")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())["cases"]
    results = []

    def record(name, ok, note, ms=None):
        results.append((name, ok, note))
        flag = "PASS" if ok is True else ("WARN" if ok is None else "FAIL")
        print(f"{flag:4} {name:34} {('%5.0f ms' % ms) if ms is not None else '       '}  {note}")

    # routes around the endpoint
    status, body, ms, _ = call("GET", base + "/v1/models", timeout=a.timeout, headers=headers)
    model = a.model
    hosted = isinstance(body, dict) and isinstance(body.get("models"), list) and body["models"] and all(
        isinstance(m, dict) and all(isinstance(m.get(k), str) for k in ("name", "description", "release_date")) for m in body["models"])
    openai = isinstance(body, dict) and isinstance(body.get("data"), list) and body["data"] and all(isinstance(m, dict) and isinstance(m.get("id"), str) for m in body["data"])
    if status == 200 and hosted:
        model = model or body["models"][0]["name"]
        record("GET /v1/models", True, f"hosted form, {len(body['models'])} model(s): {', '.join(m['name'] for m in body['models'])}", ms)
    elif status == 200 and openai:
        model = model or body["data"][0]["id"]
        record("GET /v1/models", False, "OpenAI-style 'data' list only; the hosted form is {\"models\": [{name, description, release_date}]}", ms)
    else:
        record("GET /v1/models", False, f"HTTP {status}: {str(body)[:80]}", ms)
    if not model:
        print("no model id: pass --model")
        sys.exit(2)
    status, body, ms, _ = call("GET", base + "/health", timeout=a.timeout, headers=headers)
    record("GET /health", True if status == 200 else None, f"HTTP {status}" + (" (not part of the hosted API)" if status != 200 else ""), ms)
    status, _, ms, hdrs = call("OPTIONS", base + "/v1/systemone", timeout=a.timeout)
    cors = {k.lower(): v for k, v in hdrs.items()}.get("access-control-allow-origin")
    record("OPTIONS /v1/systemone", True if status in (200, 204) and cors else None,
           f"HTTP {status}, Access-Control-Allow-Origin: {cors}" if cors else f"HTTP {status}, no CORS header (a browser page cannot call it)", ms)

    extras = set()
    for case in cases:
        request = substitute(case["request"], model)
        status, body, ms, _ = call("POST", base + "/v1/systemone", request, timeout=a.timeout, headers=headers)
        problems = []
        if case["expect"] == 200 or (case["expect"] == "4xx|200" and status == 200):
            if status != 200:
                problems.append(f"HTTP {status}: {json.dumps(body, ensure_ascii=False)[:120]}")
            else:
                check_response(request, body, problems, extras)
        else:
            if not (400 <= status < 500):
                problems.append(f"expected a 4xx, got HTTP {status}")
            elif not isinstance(body, (dict, list)):
                problems.append("error body is not JSON")
        note = "; ".join(problems) if problems else (f"HTTP {status}" + (f": {json.dumps(body, ensure_ascii=False)[:90]}" if status != 200 else ""))
        if not problems and case["expect"] == "4xx|200":
            note += "  (16-option letter readout: 422; a slot-head model answers 200)"
        record(case["name"], not problems, note, ms)
        if a.verbose and status == 200:
            print("     ", json.dumps(body, ensure_ascii=False)[:400])
    failed = [n for n, ok, _ in results if ok is False]
    warned = [n for n, ok, _ in results if ok is None]
    print(f"\n{len(results) - len(failed) - len(warned)} passed, {len(failed)} failed, {len(warned)} warnings; model {model!r} at {base}")
    if extras:
        print("extra keys beyond the hosted form (allowed):", ", ".join(sorted(extras)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
