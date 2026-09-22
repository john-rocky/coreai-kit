#!/bin/bash
# One /v1/systemone request with curl, to the model on this machine (decide-cli serve).
BASE_URL=${SYSTEM_ONE_BASE_URL:-http://127.0.0.1:8090}
curl -s "$BASE_URL/v1/systemone" -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "minicpm5-2b",
  "questions": {
    "is_urgent": {
      "type": "noul",
      "instructions": "Does this convey urgency?",
      "criteria": {"true": "explicitly time-sensitive", "false": "no urgency expressed"}
    },
    "queue": {
      "type": "choice",
      "instructions": "Which queue should handle this?",
      "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}
    }
  }
}
JSON
echo
