# fast-compact

A pi coding agent extension. Replaces the built-in compaction. When the last
request prefix is still in the model's KV cache (llama.cpp), it replays that
prefix and summarizes over it, so the prompt costs nothing. When there is no
captured request or the replay no longer fits the context window, it
serializes the summarizable slice and summarizes it cold. A progress widget
shows live token count and cache hit while it runs.

## Install

Copy `fast-compact.ts` to `~/.pi/agent/extensions/`, then `/reload` in pi.

## Use

No commands. Runs on every compaction, auto or `/compact`. The compaction
entry in the session file records `details.fastCompact` and `details.mode`
(`replay` or `slice`).

## Settings

`compaction` in `~/.pi/agent/settings.json`. Tuned values for a 128k window
(131072) with a local 27B-class model:

```json
"compaction": {
  "reserveTokens": 13312,
  "keepRecentTokens": 15360,
  "enabled": true
}
```

- `reserveTokens` is the replay-vs-slice boundary: replay while
  `tokensBefore + reserveTokens <= contextWindow`, and it is also the margin
  pi uses to trigger auto-compaction. Keep it at or above the extension's hard
  output cap (12288) plus headroom for the instruction.
- `keepRecentTokens` is the tail pi keeps verbatim after compaction. With an
  ~8k summary the post-compact context lands around 25-35k.
- `enabled` must stay true: it also gates the `/compact` command, which is
  what fires the hook. The built-in only runs if this extension errors.

Extension internals: soft summary budget 8192 (written into the prompt), hard
output cap 12288. The summary is extracted from a hidden `summarize_session`
tool call, with salvage for truncated tool-call JSON and plain-text output.
