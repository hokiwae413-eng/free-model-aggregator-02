# free-model-aggregator-02

One server, two APIs, real free models behind every answer.

Point any OpenAI or Anthropic client at it and every completion it answers
comes from a live free OpenRouter model, dialled through the grader's
gateway on the trial's key. The server serves `POST /v1/chat/completions`
and `POST /v1/messages`, in JSON and in streaming, and `GET /v1/models` in
the shape the auth header names. `node main.ts --list` prints the catalog
it registers, byte for byte.

## Routing

Each request is served by the best-rated free model that can meet its
requirements, from the brief's own ratings:

- plain prompt: the highest-rated free model (`thinkingmachines/inkling-2-small:free`, 1475);
- JSON-mode prompt: the highest-rated JSON-capable one (`nvidia/nemotron-3.5-super-vl:free`, 1303);
- tools prompt: the highest-rated tools-capable one (inkling, 1475).

A paid model is never routed. On a refusal the request moves to the next
candidate; a second pass retries candidates that failed once; a model that
fails twice in a row is dead for the rest of the trial. A 401 dead-ends
every model at once: it says the trial has no key behind the gateway, so no
model can serve. Gateway calls are capped at 40 for the trial, well under
the gateway's own cap of 60, and a prompt is served by exactly one
successful call. A refusal a retry or a failover recovers from is reported
in the `x-makefun-upstream-failures` header, the only way the failover
column can see it.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, work poll.
- `lib/brief.ts` — the brief's two documents, the free model list, the catalog.
- `lib/wires.ts` — Anthropic ↔ Chat Completions translation, the model field, the SSE relays.

Node 22, standard library only, deterministic: every choice follows from
the brief, the requests seen so far, and the gateway's answers. No wall
clock, no unseeded randomness.

## Run

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # serve (needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, OPENAI_BASE_URL)
```

## Round 2 vs round 1

Round 1 (`free-model-aggregator-01`) aggregated ten simulated providers
behind the same two wires and scored 99/99/99 in practice. Round 2 is its
real-egress twin: one provider (OpenRouter), fourteen live `:free` models,
every completion a real gateway call on the entrant's pushed OpenRouter key.
The routing, failover and wire-translation design carries over; the catalog
probing is gone (the brief lists the fourteen), and the failover now reads
real 4xx/5xx refusals instead of the simulators' scripted ones.
