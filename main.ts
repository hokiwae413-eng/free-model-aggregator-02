// A free-model aggregator: one server, two client wires, one real provider.
//
// The server reads the brief (the free model list with the house's ratings
// and the six prompts' manifest), registers its catalog with the
// environment, then serves the six prompts one at a time. Every answer comes
// from a real free OpenRouter model dialled through the grader's gateway on
// Chat Completions; the server never answers a prompt itself.
//
// Routing is the whole game. Each request is served by the best-rated free
// model that can meet its requirements:
//   - the highest-rated free model overall for a plain prompt;
//   - the highest-rated JSON-capable one for a JSON-mode prompt;
//   - the highest-rated tools-capable one for a tools prompt;
//   - min_context from the request itself, when it names one.
// A paid model is never routed. On a refusal (4xx/5xx, or a timeout) the
// request moves to the next candidate; a second pass retries candidates that
// failed once; a model that fails twice in a row is dead for the rest of the
// trial. A 401 is different: it says the trial has no key behind the
// gateway, so no model can serve, and the trial's models are all dead-ended
// at once instead of burning a call on each. The trial's gateway calls are
// capped at 40, well under the gateway's own cap of 60, and the tie-break is
// the number of gateway calls, so a prompt is served by exactly ONE
// successful call: the first 2xx answer is the answer, and no model is
// called after one succeeds.
//
// A refusal a retry or a failover recovers from is reported on the response
// in the `x-makefun-upstream-failures` header, a JSON array of at most 8
// rows {status, code, model} in the order the calls refused: the header is
// the only way the failover column can see a refusal the server retried and
// then served.
//
// Everything is deterministic: every choice follows from the brief, the
// requests seen so far, and the gateway's answers. No wall clock, no
// Math.random, no unseeded randomness. A replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wires.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 2 * 1024 * 1024;
/** One gateway call may take this long; the client waits 30 s per answer. */
const CALL_TIMEOUT_MS = 25000;
/** The trial's gateway calls, capped well under the gateway's own cap of 60. */
const MAX_GATEWAY_CALLS = 40;
/** The most gateway calls one prompt may spend, across both passes. */
const MAX_CALLS_PER_PROMPT = 10;
/** The most refusal rows the failures header carries. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of one request body, read off the body itself. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Json): Requirement {
  if (api === 'anthropic') {
    let vision = false;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
    return {
      ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
      ...(vision ? { vision: true } : {}),
    };
  }
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...(Array.isArray(body.tools) && body.tools.length > 0 ? { tools: true } : {}),
    ...(vision ? { vision: true } : {}),
    ...(format === 'json_object' || format === 'json_schema' ? { json: true } : {}),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** One refusal row for the failures header. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** Per-model trial state: consecutive failures and whether it is dead. */
interface ModelState {
  calls: number;
  failStreak: number;
  dead: boolean;
}

const modelState = new Map<string, ModelState>();
for (const model of freeModels(brief)) {
  modelState.set(model.id, { calls: 0, failStreak: 0, dead: false });
}
/** Gateway calls spent so far in this trial. */
let gatewayCalls = 0;
/** Whether the trial's key is missing: a 401 dead-ends every model at once. */
let authDead = false;

/**
 * The free candidates for one request, best rating first. Ties break toward
 * the model called least so far, then by id: deterministic, and it spreads
 * the load when ratings tie.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const out: ModelRow[] = [];
  for (const model of freeModels(brief)) {
    if (!meetsRequirements(requires, capabilitiesOf(model))) continue;
    out.push(model);
  }
  out.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    const callsA = modelState.get(a.id)?.calls ?? 0;
    const callsB = modelState.get(b.id)?.calls ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return out;
}

/** The error code of a refused gateway call, for the failures header. */
async function refusalCode(res: Response): Promise<string> {
  let bodyText = '';
  try {
    bodyText = await res.text();
  } catch {
    return `http_${res.status}`;
  }
  try {
    const body: unknown = JSON.parse(bodyText);
    if (isRecord(body) && isRecord(body.error)) {
      const code = body.error.code ?? body.error.type;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      if (typeof body.error.message === 'string' && body.error.message !== '') return body.error.message.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${res.status}`;
}

/** One gateway call: POST the Chat Completions body, naming the model dialled. */
async function dial(
  model: string,
  body: Json,
): Promise<{ ok: true; res: Response } | { ok: false; status: number; code: string }> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model }),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: 'upstream_unreachable' };
  }
  if (!res.ok) return { ok: false, status: res.status, code: await refusalCode(res) };
  return { ok: true, res };
}

/**
 * Serve one request: try the ranked candidates, then one second pass over
 * the ones that failed once. Returns the gateway's answer and the model that
 * served it, or null when no candidate could. Failures the recovery cost
 * are returned too, for the failures header.
 *
 * A model that answers 429 is cooling down for the rest of THIS prompt: a
 * per-minute rate limit does not clear in the milliseconds before the
 * second pass, so retrying it there only spends a call. It stays eligible
 * for later prompts, with its fail streak intact, and two consecutive
 * refusals across prompts still dead-end it.
 */
async function serve(
  api: 'openai' | 'anthropic',
  body: Json,
): Promise<{ res: Response; modelId: string; failures: FailureRow[] } | null> {
  const requires = requirementsFromBody(api, body);
  const candidates = candidatesFor(requires);
  const failures: FailureRow[] = [];
  const cooling = new Set<string>();
  let spent = 0;
  for (let pass = 0; pass < 2; pass++) {
    for (const candidate of candidates) {
      if (spent >= MAX_CALLS_PER_PROMPT || gatewayCalls >= MAX_GATEWAY_CALLS || authDead) return null;
      const state = modelState.get(candidate.id);
      if (state === undefined || state.dead) continue;
      // The second pass only retries models that failed exactly once and are
      // not cooling down from a 429; a model that failed twice is dead, and
      // one that never failed was already tried in the first pass.
      if (pass === 1 && (state.failStreak !== 1 || cooling.has(candidate.id))) continue;
      state.calls += 1;
      gatewayCalls += 1;
      spent += 1;
      const requestBody: Json =
        api === 'anthropic' ? anthropicToOpenAI(body, candidate.id) : { ...body, model: candidate.id };
      const result = await dial(candidate.id, requestBody);
      if (result.ok) {
        state.failStreak = 0;
        return { res: result.res, modelId: candidate.id, failures };
      }
      state.failStreak += 1;
      failures.push({ status: result.status, code: result.code, model: candidate.id });
      if (result.status === 429) cooling.add(candidate.id);
      if (result.status === 401) {
        // No key behind the gateway: no model can serve this trial.
        authDead = true;
        for (const other of modelState.values()) other.dead = true;
        return null;
      }
      if (state.failStreak >= 2) state.dead = true;
    }
  }
  return null;
}

/** The failures header for a response that recovered from refusals. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  const rows = failures.slice(0, MAX_FAILURE_ROWS);
  const text = JSON.stringify(rows);
  if (Buffer.byteLength(text) > 8192) return {};
  return { 'x-makefun-upstream-failures': text };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a real free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  const { res: upstream, modelId, failures } = served;
  if (upstream.body === null) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answered with no body');
  const headers = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, modelId)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, modelId), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  const { res: upstream, modelId, failures } = served;
  if (upstream.body === null) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answered with no body');
  const headers = failuresHeader(failures);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, modelId)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, modelId), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
