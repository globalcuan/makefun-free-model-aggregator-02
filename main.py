"""Run a deterministic server for free models from a competition brief.
Translate OpenAI and Anthropic requests through an OpenAI chat gateway.
Register the catalog and stop after the environment finishes its work.
"""
import http.server
import json
import os
import re
import socketserver
import threading
import time
import urllib.error
import urllib.request

MODELS = []
PROMPTS = []
CATALOG = {"providers": []}
CATALOG_TEXT = '{"providers":[]}'
RESPONSE_ID = 0
SERVED = 0
INFLIGHT = 0
LAST_FINISHED = time.monotonic()
DONE = False
DONE_AT = None
LOCK = threading.RLock()
EXITING = threading.Event()


def load_brief(path):
    raw = open(path, "r", encoding="utf-8").read()
    dec = json.JSONDecoder()
    docs = {}
    for marker, schema in (("free-model-aggregator-02/models/v1", "models"),
                           ("free-model-aggregator-02/prompts/v1", "prompts")):
        pos = raw.find('"schema"')
        while pos >= 0:
            start = raw.rfind("{", 0, pos + 1)
            try:
                obj, _ = dec.raw_decode(raw[start:])
            except (ValueError, TypeError):
                pos = raw.find('"schema"', pos + 1)
                continue
            if obj.get("schema") == marker:
                docs[schema] = obj
                break
            pos = raw.find('"schema"', pos + 1)
    return docs.get("models", {}).get("models", []), docs.get("prompts", {}).get("prompts", [])


def build_catalog():
    global MODELS, PROMPTS, CATALOG, CATALOG_TEXT
    path = os.environ.get("MAKEFUN_PROMPT_FILE")
    if path:
        MODELS, PROMPTS = load_brief(path)
    seen, catalog_models = set(), []
    for m in MODELS:
        mid = m.get("id")
        if m.get("free") is True and mid not in seen:
            seen.add(mid)
            catalog_models.append({"id": mid, "free": True, "context_length": m.get("context_length")})
    CATALOG = {"providers": []}
    if catalog_models:
        CATALOG["providers"].append({"id": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "key_env": "OPENROUTER_API_KEY", "wire": "openai", "models": catalog_models})
    CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def json_bytes(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_json(req):
    return json.loads(req.data.decode("utf-8")) if req.data else {}


def post_json(url, body, headers=None, timeout=10):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=json_bytes(body), headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def get_work():
    req = urllib.request.Request(os.environ["MAKEFUN_ENV_URL"].rstrip("/") + "/work")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def poller():
    global DONE, DONE_AT
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    while not EXITING.is_set():
        try:
            result = get_work()
            WORK_NEXT[0] = result.get("next")
            if result.get("done"):
                if not DONE:
                    DONE = True
                    DONE_AT = time.monotonic()
                with LOCK:
                    if INFLIGHT == 0 and time.monotonic() - max(LAST_FINISHED, DONE_AT) >= 0.5:
                        os._exit(0)
        except Exception:
            pass
        time.sleep(0.1)


def requirements_for():
    global SERVED
    next_id = getattr(threading.current_thread(), "prompt_id", None)
    # Work identifiers are refreshed by the poll thread through WORK_NEXT.
    next_id = WORK_NEXT[0] or next_id
    if next_id:
        for p in PROMPTS:
            if p.get("id") == next_id:
                return p.get("requires", {}) or {}
    if SERVED < len(PROMPTS):
        return PROMPTS[SERVED].get("requires", {}) or {}
    return {}


WORK_NEXT = [None]


def candidates(reqs):
    def meets(m):
        if m.get("free") is not True:
            return False
        for flag in ("tools", "vision", "json"):
            if reqs.get(flag) and not m.get(flag):
                return False
        return m.get("context_length", 0) >= reqs.get("min_context", 0)
    return sorted((m for m in MODELS if meets(m)), key=lambda m: -m.get("rating_milli", 0))


def anthropic_to_openai(body):
    messages = []
    if body.get("system"):
        system = body["system"]
        if isinstance(system, list):
            system = "".join(x.get("text", "") for x in system if x.get("type") == "text")
        messages.append({"role": "system", "content": system})
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content}); continue
        parts = []; tool_calls = []
        for block in content:
            typ = block.get("type")
            if typ == "text": parts.append({"type": "text", "text": block.get("text", "")})
            elif typ == "image":
                src = block.get("source", {})
                if src.get("type") == "base64":
                    parts.append({"type": "image_url", "image_url": {"url": "data:" + src.get("media_type", "application/octet-stream") + ";base64," + src.get("data", "")}})
                elif src.get("type") == "url": parts.append({"type": "image_url", "image_url": {"url": src.get("url")}})
            elif typ == "tool_use":
                tool_calls.append({"id": block.get("id", ""), "type": "function", "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}), separators=(",", ":"))}})
            elif typ == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list): result = "".join(b.get("text", "") for b in result if b.get("type") == "text")
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": result})
        if parts or tool_calls:
            item = {"role": role}
            if parts: item["content"] = parts
            if tool_calls: item["tool_calls"] = tool_calls
            messages.append(item)
    out = {k: body[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "stream", "stop_sequences", "tools", "tool_choice", "metadata") if k in body}
    out["messages"] = messages
    if "max_tokens" in out: pass
    if "stop_sequences" in out: out["stop"] = out.pop("stop_sequences")
    if "tools" in out:
        out["tools"] = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("input_schema", {})}} for t in out["tools"]]
    choice = out.get("tool_choice")
    if isinstance(choice, dict):
        typ = choice.get("type")
        out["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}} if typ == "tool" else typ
    return out


def stream_collect(response):
    text = ""; calls = {}; finish = None; usage = {}
    for raw in response.read().decode("utf-8", "replace").splitlines():
        if not raw.startswith("data:"): continue
        payload = raw[5:].strip()
        if payload == "[DONE]": continue
        try: chunk = json.loads(payload)
        except ValueError: continue
        if chunk.get("usage"): usage = chunk["usage"]
        for c in chunk.get("choices", []):
            delta = c.get("delta", {})
            text += delta.get("content") or ""
            for call in delta.get("tool_calls", []):
                i = call.get("index", 0); cur = calls.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"): cur["id"] = call["id"]
                fn = call.get("function", {}); cur["function"]["name"] += fn.get("name", ""); cur["function"]["arguments"] += fn.get("arguments", "")
            if c.get("finish_reason") is not None: finish = c["finish_reason"]
    return text, [calls[i] for i in sorted(calls)], finish, usage


def normalize_answer(text, calls, finish, usage, wire):
    stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}.get(finish, finish or "end_turn")
    if calls: stop = "tool_use"
    if wire == "openai": stop = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}.get(stop, stop)
    return {"text": text, "tool_calls": calls, "stop": stop,
            "input": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output": usage.get("completion_tokens", usage.get("output_tokens", 0))}


def openai_error(message):
    return {"error": {"message": message, "type": "server_error", "param": None, "code": None}}


def client_error(message, wire):
    if wire == "anthropic":
        return {"type": "error", "error": {"type": "api_error", "message": message}}
    return openai_error(message)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_): pass
    def send_obj(self, obj, status=200):
        data = json_bytes(obj); self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path != "/v1/models": return self.send_obj(openai_error("Not found"), 404)
        free_ids = [m["id"] for m in CATALOG["providers"][0]["models"]] if CATALOG["providers"] else []
        by_id = {m["id"]: m for m in MODELS}
        if self.headers.get("x-api-key") is not None:
            data = [{"type":"model", "id":mid, "display_name":by_id[mid].get("display_name",mid), "created_at":"2026-01-01T00:00:00Z"} for mid in free_ids]
            return self.send_obj({"data": data, "has_more": False})
        data = [{"id":mid,"object":"model", "created":1767225600,"owned_by":"openrouter"} for mid in free_ids]
        self.send_obj({"object":"list","data":data})
    def do_POST(self):
        global INFLIGHT, LAST_FINISHED, RESPONSE_ID, SERVED
        wire = "anthropic" if self.path == "/v1/messages" else "openai"
        if self.path not in ("/v1/chat/completions", "/v1/messages"):
            return self.send_obj(openai_error("Not found"), 404)
        with LOCK: INFLIGHT += 1
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            original_stream = bool(body.get("stream"))
            reqs = requirements_for(); failed = set(); answer = None; model_id = None
            source = anthropic_to_openai(body) if wire == "anthropic" else dict(body)
            for model in candidates(reqs):
                if model["id"] in failed: continue
                model_id = model["id"]; upstream = dict(source); upstream["model"] = model_id
                if original_stream: upstream["stream_options"] = {"include_usage": True}
                base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
                req = urllib.request.Request(base + "/chat/completions", data=json_bytes(upstream), method="POST",
                    headers={"Content-Type":"application/json", "Authorization":"Bearer " + os.environ.get("OPENAI_API_KEY", "")})
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        if original_stream:
                            text, calls, finish, usage = stream_collect(resp)
                        else:
                            result = json.loads(resp.read().decode("utf-8")); choice = result.get("choices", [{}])[0]
                            msg = choice.get("message", {}); text = msg.get("content") or ""; calls = msg.get("tool_calls", [])
                            finish = choice.get("finish_reason"); usage = result.get("usage", {})
                    answer = normalize_answer(text, calls, finish, usage, wire); break
                except urllib.error.HTTPError:
                    failed.add(model_id); continue
                except Exception:
                    failed.add(model_id); continue
            if answer is None:
                return self.send_obj(client_error("All free models failed", wire), 502)
            with LOCK: RESPONSE_ID += 1; rid = RESPONSE_ID
            if original_stream: self.send_sse(answer, model_id, wire, rid)
            else: self.send_obj(self.render_json(answer, model_id, wire, rid))
            SERVED += 1
        except Exception as exc:
            self.send_obj(client_error(str(exc), wire), 502)
        finally:
            with LOCK:
                INFLIGHT -= 1; LAST_FINISHED = time.monotonic()
    def render_json(self, a, model, wire, rid):
        if wire == "openai":
            msg = {"role":"assistant", "content":a["text"] or None}
            if a["tool_calls"]: msg["tool_calls"] = a["tool_calls"]
            return {"id":f"chatcmpl-{rid}","object":"chat.completion","created":1767225600,"model":model,
                "choices":[{"index":0,"message":msg,"finish_reason":a["stop"]}],
                "usage":{"prompt_tokens":a["input"],"completion_tokens":a["output"],"total_tokens":a["input"]+a["output"]}}
        content=[]
        if a["text"]: content.append({"type":"text","text":a["text"]})
        for c in a["tool_calls"]:
            fn=c.get("function",{}); 
            try: inp=json.loads(fn.get("arguments") or "{}")
            except ValueError: inp={}
            content.append({"type":"tool_use","id":c.get("id"),"name":fn.get("name"),"input":inp})
        return {"id":f"msg_{rid}","type":"message","role":"assistant","model":model,"content":content,
            "stop_reason":a["stop"],"stop_sequence":None,"usage":{"input_tokens":a["input"],"output_tokens":a["output"]}}
    def send_sse(self, a, model, wire, rid):
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Cache-Control","no-cache"); self.send_header("Connection","close"); self.end_headers()
        def emit(obj, event=None):
            if event: self.wfile.write(("event: "+event+"\n").encode())
            self.wfile.write(b"data: "+json_bytes(obj)+b"\n\n")
        if wire == "openai":
            emit({"id":f"chatcmpl-{rid}","object":"chat.completion.chunk","created":1767225600,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})
            for i in range(0,len(a["text"]),40): emit({"id":f"chatcmpl-{rid}","object":"chat.completion.chunk","created":1767225600,"model":model,"choices":[{"index":0,"delta":{"content":a["text"][i:i+40]},"finish_reason":None}]})
            if a["tool_calls"]: emit({"id":f"chatcmpl-{rid}","object":"chat.completion.chunk","created":1767225600,"model":model,"choices":[{"index":0,"delta":{"tool_calls":a["tool_calls"]},"finish_reason":None}]})
            emit({"id":f"chatcmpl-{rid}","object":"chat.completion.chunk","created":1767225600,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":a["stop"]}]})
            emit({"id":f"chatcmpl-{rid}","object":"chat.completion.chunk","created":1767225600,"model":model,"choices":[],"usage":{"prompt_tokens":a["input"],"completion_tokens":a["output"],"total_tokens":a["input"]+a["output"]}})
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            emit({"type":"message_start","message":{"id":f"msg_{rid}","type":"message","role":"assistant","model":model,"content":[],"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":a["input"],"output_tokens":0}}},"message_start")
            idx=0
            if a["text"]:
                emit({"type":"content_block_start","index":idx,"content_block":{"type":"text","text":""}},"content_block_start")
                for i in range(0,len(a["text"]),40): emit({"type":"content_block_delta","index":idx,"delta":{"type":"text_delta","text":a["text"][i:i+40]}},"content_block_delta")
                emit({"type":"content_block_stop","index":idx},"content_block_stop"); idx+=1
            for c in a["tool_calls"]:
                fn=c.get("function",{}); emit({"type":"content_block_start","index":idx,"content_block":{"type":"tool_use","id":c.get("id"),"name":fn.get("name"),"input":{}}},"content_block_start")
                emit({"type":"content_block_delta","index":idx,"delta":{"type":"input_json_delta","partial_json":fn.get("arguments","")}},"content_block_delta")
                emit({"type":"content_block_stop","index":idx},"content_block_stop"); idx+=1
            emit({"type":"message_delta","delta":{"stop_reason":a["stop"],"stop_sequence":None},"usage":{"output_tokens":a["output"]}},"message_delta")
            emit({"type":"message_stop"},"message_stop")
        self.wfile.flush()


def main():
    build_catalog()
    if "--list" in os.sys.argv:
        print(CATALOG_TEXT); return 0
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    post_json(env_url + "/register", {"protocol_version":1,"port":server.server_address[1],"catalog":CATALOG})
    threading.Thread(target=poller, daemon=True).start()
    try:
        while True: time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
