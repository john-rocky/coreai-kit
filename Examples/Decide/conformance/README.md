# Conformance and calibration

Two standard-library Python scripts, for this server or any other that speaks `/v1/systemone`.

```bash
python3 check.py http://127.0.0.1:8090              # 22 requests (cases.json) + the three routes around them
python3 calibration.py authored144.jsonl minicpm5-2b.jsonl [--applied 1.03] [--bins 10]
```

`check.py` sends every request in `cases.json` in the hosted System One forms — the three
question types, the 16-, 17-, 255- and 256-option edges, 2 and 10 score levels, structured
state and instructions, a chat as the state, four malformed requests — and checks the status
and the shape of each answer (keys and types, never values). Extra keys are listed, not failed.
`GET /v1/models` is checked against the hosted list form.

`calibration.py` reads a fixture and `decide-cli oracle` output and prints accuracy, family
balanced accuracy, NLL, Brier and top-label ECE per family, then the temperature that
minimises NLL on those rows. Results for the catalog models are in
[`../README.md`](../README.md#measured).
