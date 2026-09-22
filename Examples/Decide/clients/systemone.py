#!/usr/bin/env python3
"""A /v1/systemone request to the model on this machine, with nothing but the standard library.

    decide-cli serve                       # in another terminal
    python3 clients/systemone.py           # runs the example below
    SYSTEM_ONE_BASE_URL=http://127.0.0.1:8090 python3 clients/systemone.py

The request and answer forms are the hosted System One ones, so a client library written for
the hosted endpoint works too: give it this base URL (asynq-io/system-one reads
SYSTEM_ONE_BASE_URL; jev-rs-style clients read TYPESAFE_BASE_URL) and any API key.
"""
import json
import os
import sys
import urllib.request

BASE_URL = os.environ.get("SYSTEM_ONE_BASE_URL", "http://127.0.0.1:8090")


def decide(state, questions, model="minicpm5-2b"):
    body = json.dumps({"state": state, "model": model, "questions": questions}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request) as response:
        return json.load(response)


if __name__ == "__main__":
    state = sys.argv[1] if len(sys.argv) > 1 else "Help! My payouts have been failing for 3 days. Refund the duplicate charge today or we cancel."
    answers = decide(state, {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                      "criteria": {"true": "explicitly time-sensitive", "false": "no urgency expressed"}},
        "queue": {"type": "choice", "instructions": "Which queue should handle this?",
                  "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}},
        "anger": {"type": "score", "instructions": "How upset is the customer?",
                  "criteria": ["calm", "frustrated", "very angry"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to leave?"},
    })
    print(json.dumps(answers, indent=2, ensure_ascii=False))
