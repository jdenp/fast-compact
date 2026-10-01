import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertToLlm, estimateTokens } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";

// Replacement for the built-in compaction: the handler always produces the
// compaction result itself. Two paths:
// - replay: the captured prefix of the last request is replayed; llama.cpp keeps
//   it in its KV cache, so the summarization call costs only the instruction
//   tail plus the streamed summary, not a cold re-read of the conversation.
// - slice: no captured request (fresh restart) or the replay no longer fits in
//   the context window: serialize the summarizable slice of the session and
//   summarize it cold. Same shape as pi's built-in serializer.
// The replay-vs-slice boundary is the physical fit of the replay request:
// captured prefix + instruction tail + MAX_OUTPUT must fit the context window.
// Replay while that fits; the serialized slice is the cold fallback.

const TOOL_NAME = "summarize_session";
const SUMMARY_BUDGET = 8192; // soft token cap, written into the prompt
const MAX_OUTPUT = 12288; // hard cap on the summarization stream
const REPLAY_TAIL_MARGIN = 512; // instruction user message + chat template overhead, in tokens
const TOOL_RESULT_MAX_CHARS = 2000; // per-tool-result truncation in slice serialization (matches pi's built-in)
const WIDGET_KEY = "fast-compact";
const PAINT_MS = 150;
const BAR_WIDTH = 24;
const FINAL_HOLD_MS = 2500;

const SKELETON = `## Goal
[one or two sentences: what the user is trying to accomplish]

## Constraints & Preferences
- [explicit constraints, preferences, and repo conventions; "(none)" if empty]

## Progress
### Done
- [x] [completed work items]
### In Progress
- [ ] [what is happening now]
### Blocked
- [blockers and why; "(none)" if empty]

## Key Decisions
- **[decision]**: [why it was made]

## Next Steps
1. [ordered concrete next actions]

## Critical Context
- [exact file paths, function names, commands, and error messages a continuation session must not re-derive; "(none)" if empty]`;

function buildPrompt(sourceNote: string, customInstructions?: string): string {
  const lines = [
    "This is a session compaction checkpoint. Call the summarize_session tool immediately, with exactly one argument named summary.",
    sourceNote,
    "",
    "Rules:",
    "- summary is a structured markdown checkpoint of this ENTIRE session. A fresh LLM must be able to continue the work from the summary alone.",
    "- If the conversation contains an earlier compaction summary (a user message starting with \"The conversation history before this point was compacted\"), merge all still-relevant information from it into the new summary.",
    "- Keep exact file paths, function names, command lines, and error messages verbatim.",
    `- Keep the summary under ${SUMMARY_BUDGET} tokens.`,
    "- Use this exact format:",
    "",
    SKELETON,
  ];
  if (customInstructions) {
    lines.push("", `Additional focus for this compaction: ${customInstructions}`);
  }
  lines.push("", "Do not call any other tool. Do not write text outside the tool call.");
  // /no_think: replay sends enable_thinking=true to match the cached rendering; the marker stops the model thinking.
  return lines.join("\n") + "\n/no_think";
}

function fmtTok(n: number): string {
  if (n < 1000) return String(n);
  if (n < 100000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return `${Math.round(n / 1000)}k`;
}

function bar(fraction: number): string {
  const filled = Math.round(Math.max(0, Math.min(1, fraction)) * BAR_WIDTH);
  return "\u2593".repeat(filled) + "\u2591".repeat(BAR_WIDTH - filled);
}

// The summary may arrive as a complete tool call, a truncated JSON string (length
// stop), or plain text. Try each in order.
function salvageSummary(toolJson: string): string | undefined {
  try {
    const parsed = JSON.parse(toolJson) as { summary?: unknown };
    if (typeof parsed.summary === "string" && parsed.summary.trim()) return parsed.summary.trim();
  } catch {
    // truncated JSON, fall through
  }
  const i = toolJson.indexOf('"summary"');
  if (i === -1) return undefined;
  const rest = toolJson.slice(i);
  const m = rest.match(/:\s*"/);
  if (!m || m.index === undefined) return undefined;
  let out = "";
  let j = m.index + m[0].length;
  while (j < rest.length) {
    const ch = rest[j];
    if (ch === "\\") {
      const n = rest[j + 1];
      if (n === "n") out += "\n";
      else if (n === "t") out += "\t";
      else if (n === "r") out += "\r";
      else if (n === '"') out += '"';
      else if (n === "\\") out += "\\";
      else if (n === "/") out += "/";
      else if (n === "u") {
        const hex = rest.slice(j + 2, j + 6);
        if (/^[0-9a-fA-F]{4}$/.test(hex)) {
          out += String.fromCharCode(parseInt(hex, 16));
          j += 6;
          continue;
        }
        out += n;
      } else out += n;
      j += 2;
    } else if (ch === '"') {
      break;
    } else {
      out += ch;
      j += 1;
    }
  }
  return out.trim() || undefined;
}

// Slice-path serialization: the same shapes pi's built-in summarizer uses.
function contentTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((b): b is { text: string } => (b as { type?: unknown }).type === "text" && typeof (b as { text?: unknown }).text === "string")
    .map((b) => b.text)
    .join(" ");
}

function serializeConversation(messages: unknown[]): string {
  const parts: string[] = [];
  for (const raw of messages) {
    const msg = raw as { role: string; content?: unknown; name?: string; arguments?: Record<string, unknown> };
    if (msg.role === "user") {
      const text = contentTextOf(msg.content);
      if (text.trim()) parts.push(`[User]: ${text}`);
    } else if (msg.role === "assistant") {
      const blocks = Array.isArray(msg.content)
        ? (msg.content as { type: string; text?: string; name?: string; arguments?: Record<string, unknown> }[])
        : [];
      let text = "";
      const calls: string[] = [];
      for (const block of blocks) {
        if (block.type === "text" && block.text) text += block.text;
        else if (block.type === "toolCall" && block.name) {
          const argsStr = Object.entries(block.arguments ?? {})
            .map(([k, v]) => `${k}=${JSON.stringify(v) ?? "undefined"}`)
            .join(", ");
          calls.push(`${block.name}(${argsStr})`);
        }
      }
      if (text.trim()) parts.push(`[Assistant]: ${text}`);
      if (calls.length) parts.push(`[Assistant tool calls]: ${calls.join("; ")}`);
    } else if (msg.role === "toolResult") {
      const text = contentTextOf(msg.content);
      if (text.trim()) {
        const out = text.length > TOOL_RESULT_MAX_CHARS ? `${text.slice(0, TOOL_RESULT_MAX_CHARS)}... [truncated]` : text;
        parts.push(`[Tool result]: ${out}`);
      }
    }
  }
  return parts.join("\n\n");
}

export default function fastCompact(pi: ExtensionAPI) {
  let captured: AgentMessage[] | undefined;
  let finalTimer: NodeJS.Timeout | undefined;

  pi.registerTool({
    name: TOOL_NAME,
    label: "Summarize Session",
    description:
      "Write a structured markdown checkpoint of the whole session so far. Called by the compaction system with one argument: summary.",
    parameters: Type.Object({
      summary: Type.String({
        description: "Structured markdown checkpoint of the session, in the exact section format given in the message.",
      }),
    }),
    execute: async () => {
      // Only the compaction path consumes this tool (it reads the stream, pi never
      // executes it). A spontaneous in-session call is harmless.
      return { content: [{ type: "text", text: "Session summary recorded (internal compaction tool)." }], details: {} };
    },
  });

  const hide = (ctx: { hasUI?: boolean; ui?: { setWidget?: (key?: string, lines?: string[] | undefined) => void } }) => {
    if (ctx.hasUI) ctx.ui?.setWidget?.(WIDGET_KEY, undefined);
  };

  // A stale capture from another session would replay the wrong prefix and
  // summarize the wrong conversation, so drop it on every session transition.
  pi.on("session_start", (_event, ctx) => {
    captured = undefined;
    hide(ctx);
  });
  pi.on("session_shutdown", (_event, ctx) => {
    captured = undefined;
    hide(ctx);
    if (finalTimer) {
      clearTimeout(finalTimer);
      finalTimer = undefined;
    }
  });

  pi.on("context_with_system", (event) => {
    // The runner hands handlers a structuredClone of the final transcript, so
    // holding the reference is safe and it is exactly what the provider sends.
    captured = event.messages;
  });

  pi.on("session_before_compact", async (event, ctx) => {
    const { preparation, customInstructions, signal } = event;
    if (!ctx.model) return; // no model to stream with; let the built-in try
    // The replay request sends the captured prefix, not the current context, so fit is tested on the prefix
    const prefixTok = captured ? captured.reduce((n, m) => n + estimateTokens(m), 0) : 0;
    const canReplay =
      !!captured && prefixTok + REPLAY_TAIL_MARGIN + MAX_OUTPUT <= ctx.model.contextWindow;
    if (!canReplay && preparation.messagesToSummarize.length === 0 && preparation.turnPrefixMessages.length === 0)
      return; // nothing to summarize either way

    const show = (line: string) => {
      if (ctx.hasUI) ctx.ui.setWidget(WIDGET_KEY, [line]);
    };

    // Gauge: a checkpoint usually lands around 1/30 of the input it covers, so
    // predict the summary length per run instead of measuring against the 4k cap.
    // Qwen tokenizes ~2x denser than pi's chars/4 house estimate.
    const charsPerTok = /qwen/i.test(ctx.model.id) ? 2 : 4;
    let messages: { role: string; content: unknown; timestamp?: number }[];
    let scopeTok: number;
    if (canReplay) {
      scopeTok = preparation.tokensBefore;
      messages = [
        ...convertToLlm(captured),
        { role: "user" as const, content: [{ type: "text" as const, text: buildPrompt("The transcript above is the full session.", customInstructions) }], timestamp: Date.now() },
      ];
    } else {
      const conversation = serializeConversation(
        convertToLlm([...preparation.turnPrefixMessages, ...preparation.messagesToSummarize]),
      );
      scopeTok = Math.round(conversation.length / charsPerTok);
      const text = `${buildPrompt("The <conversation> block below is the session history to be summarized.", customInstructions)}\n\n<conversation>\n${conversation}\n</conversation>`;
      messages = [{ role: "user" as const, content: [{ type: "text" as const, text }], timestamp: Date.now() }];
    }
    const expectedTok = Math.max(512, Math.min(SUMMARY_BUDGET, Math.round(scopeTok / 30)));

    const startedAt = Date.now();
    let outChars = 0;
    let lastPaint = 0;
    const paint = (force: boolean) => {
      const now = Date.now();
      if (!force && now - lastPaint < PAINT_MS) return;
      lastPaint = now;
      const tok = Math.round(outChars / charsPerTok);
      const frac = Math.min(1, tok / expectedTok);
      show(
        `Compacting: ${bar(frac)} ${Math.round(frac * 100)}%  ${fmtTok(tok)}/~${fmtTok(expectedTok)} tok  ${((now - startedAt) / 1000).toFixed(1)}s`,
      );
    };
    show(
      canReplay
        ? "Compacting: waiting for first token (replaying cached prefix)"
        : "Compacting: waiting for first token (serialized slice, cold)",
    );

    let summary: string | undefined;
    let toolJson = "";
    let textOut = "";
    let usage: AssistantMessage["usage"];
    let truncated = false;
    let aborted = false;
    let streamError: string | undefined;

    try {
      // Same thinking flag as the cached request, or the template restamps the prefix and the cache misses.
      const stream = ctx.modelRegistry.streamSimple(ctx.model, { messages }, {
        maxTokens: MAX_OUTPUT,
        signal,
        sessionId: ctx.sessionManager.getSessionId(),
        reasoning: ctx.thinkingLevel,
      });
      for await (const ev of stream) {
        switch (ev.type) {
          case "toolcall_delta":
            toolJson += ev.delta;
            outChars += ev.delta.length;
            paint(false);
            break;
          case "text_delta":
            textOut += ev.delta;
            outChars += ev.delta.length;
            paint(false);
            break;
          case "toolcall_end":
            if (ev.toolCall.name === TOOL_NAME && typeof ev.toolCall.arguments.summary === "string") {
              summary = (ev.toolCall.arguments.summary as string).trim();
            }
            break;
          case "done":
            usage = ev.message.usage;
            truncated = ev.reason === "length";
            break;
          case "error":
            aborted = ev.reason === "aborted";
            streamError = aborted ? "aborted" : (ev.error.errorMessage ?? "stream error");
            break;
        }
      }
    } catch (err) {
      streamError = err instanceof Error ? err.message : String(err);
    }

    if (aborted || signal.aborted) {
      hide(ctx);
      return { cancel: true };
    }
    if (streamError) {
      hide(ctx);
      if (ctx.hasUI) ctx.ui.notify(`fast-compact: ${streamError}; falling back to built-in compaction`, "warning");
      return;
    }
    if (!summary) summary = salvageSummary(toolJson);
    // a narrated fake tool call is not a checkpoint; only accept prose with the skeleton header
    if (!summary && textOut.includes("## Goal")) summary = textOut.trim();
    if (!summary) {
      hide(ctx);
      if (ctx.hasUI) ctx.ui.notify("fast-compact: model returned no summary; falling back to built-in compaction", "warning");
      return;
    }
    const mode = canReplay ? "replay" : "slice";

    const outTok = usage?.output ?? Math.round(outChars / 4);
    const after = preparation.settings.keepRecentTokens + outTok;
    const secs = ((Date.now() - startedAt) / 1000).toFixed(1);
    const promptTok = (usage?.input ?? 0) + (usage?.cacheRead ?? 0);
    const hitPct = promptTok > 0 ? Math.round((100 * (usage?.cacheRead ?? 0)) / promptTok) : -1;
    const cacheNote = hitPct < 0 ? "cache n/a" : `cache ${hitPct}%`;
    show(
      `Compacted: ${fmtTok(outTok)} tok summary in ${secs}s  ${cacheNote}  context ${fmtTok(preparation.tokensBefore)} -> ~${fmtTok(after)}${truncated ? " (truncated)" : ""}`,
    );
    if (mode === "replay" && hitPct >= 0 && hitPct < 50 && ctx.hasUI)
      ctx.ui.notify(`fast-compact: prefix cache hit only ${hitPct}% - replay ran mostly cold (prefix changed or evicted?)`, "warning");
    if (finalTimer) clearTimeout(finalTimer);
    finalTimer = setTimeout(() => {
      hide(ctx);
      finalTimer = undefined;
    }, FINAL_HOLD_MS);

    return {
      compaction: {
        summary,
        firstKeptEntryId: preparation.firstKeptEntryId,
        tokensBefore: preparation.tokensBefore,
        estimatedTokensAfter: after,
        usage,
        details: {
          fastCompact: true,
          mode,
          truncated,
          readFiles: [...preparation.fileOps.read],
          modifiedFiles: [...preparation.fileOps.written, ...preparation.fileOps.edited],
        },
      },
    };
  });
}
