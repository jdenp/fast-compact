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

## Notes

The replay-vs-slice boundary is `compaction.reserveTokens` in
`~/.pi/agent/settings.json` (fallback 12288 when unset): replay while
`tokensBefore + reserveTokens <= contextWindow`. Keep that setting at or above
the extension's hard output cap (8192) plus headroom.

The summary is extracted from a hidden `summarize_session` tool call, with
salvage for truncated tool-call JSON and plain-text output.
