"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Tokenizer } from "@huggingface/tokenizers";


const models = [
  { id: "google/umt5-xxl", label: "UmT5 XXL (Google)", maxContext: 500 },
];

const LAYER_COLORS = ["#f97316", "#a855f7", "#06b6d4", "#84cc16", "#ec4899", "#eab308", "#3b82f6", "#ef4444"];

const normalizeRanges = (ranges) => {
  const sorted = ranges.filter(([s, e]) => e > s).map(([s, e]) => [s, e]).sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of sorted) {
    if (merged.length && r[0] <= merged[merged.length - 1][1]) {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r[1]);
    } else {
      merged.push([r[0], r[1]]);
    }
  }
  return merged;
};

const subtractRange = (ranges, [ss, se]) => {
  const out = [];
  for (const [s, e] of ranges) {
    if (e <= ss || s >= se) { out.push([s, e]); continue; }
    if (s < ss) out.push([s, ss]);
    if (e > se) out.push([se, e]);
  }
  return out;
};

const clampRanges = (ranges, max) =>
  ranges.map(([s, e]) => [Math.min(s, max), Math.min(e, max)]).filter(([s, e]) => e > s);

const snapRangeToTokens = ([start, end], offsets, promptLength) => {
  if (!offsets.length) return [start, end];
  const boundaries = new Set([0, promptLength]);
  for (const [s, e] of offsets) {
    boundaries.add(s);
    boundaries.add(e);
  }
  const sorted = [...boundaries].sort((a, b) => a - b);
  let snappedStart = 0;
  let snappedEnd = promptLength;
  for (const b of sorted) {
    if (b <= start) snappedStart = b;
    if (b >= end) { snappedEnd = b; break; }
  }
  if (snappedEnd <= snappedStart) return null;
  return [snappedStart, snappedEnd];
};

const buildHighlightSegments = (text, layers) => {
  const all = [];
  for (const l of layers) {
    for (const [s, e] of l.ranges) all.push({ s, e, color: l.color });
  }
  all.sort((a, b) => a.s - b.s);
  const out = [];
  let cur = 0;
  for (const r of all) {
    if (r.s > cur) out.push({ text: text.slice(cur, r.s), color: null });
    out.push({ text: text.slice(r.s, r.e), color: r.color });
    cur = Math.max(cur, r.e);
  }
  if (cur < text.length) out.push({ text: text.slice(cur), color: null });
  return out;
};

const extractSpecialIds = (tokenizerConfig) => {
  const specials = [];
  const st = tokenizerConfig?.special_tokens || {};
  Object.values(st).forEach((val) => {
    if (val && typeof val.id === "number") specials.push(val.id);
    if (val && typeof val.id === "string") specials.push(Number(val.id));
  });
  return new Set(specials.filter((n) => Number.isFinite(n)));
};


export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [modelId, setModelId] = useState(models[0].id);
  const [hfToken, setHfToken] = useState("");
  const [status, setStatus] = useState({ text: "Idle", kind: "idle" });
  const [summary, setSummary] = useState({ vocab: "—", tokens: "—", specials: "—" });
  const [rows, setRows] = useState([]);
  const [modelOpen, setModelOpen] = useState(false);
  const [tokenOpen, setTokenOpen] = useState(false);
  const [maxOpen, setMaxOpen] = useState(false);
  const [streamOpen, setStreamOpen] = useState(false);
  const [maxContext, setMaxContext] = useState(models[0].maxContext);
  const [layers, setLayers] = useState([]);
  const [activeLayerId, setActiveLayerId] = useState(null);
  const [activeMode, setActiveMode] = useState("edit");
  const [lastTokenizedModel, setLastTokenizedModel] = useState(null);
  const [tokenOffsets, setTokenOffsets] = useState([]);
  const [ratioMode, setRatioMode] = useState("context");
  const layerIdRef = useRef(1);

  const cacheRef = useRef(new Map());
  const modelFieldRef = useRef(null);
  const promptRef = useRef(null);

  useEffect(() => {
    const ta = promptRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${ta.scrollHeight}px`;
  }, [prompt]);

  const tokenizeRef = useRef(null);
  useEffect(() => {
    if (!prompt.trim()) {
      setRows([]);
      setTokenOffsets([]);
      setSummary((s) => ({ ...s, tokens: "—" }));
      return;
    }
    const id = setTimeout(() => {
      tokenizeRef.current?.();
    }, 250);
    return () => clearTimeout(id);
  }, [prompt, modelId, hfToken]);

  useEffect(() => {
    const preset = models.find((m) => m.id === modelId);
    if (preset) setMaxContext(preset.maxContext);
  }, [modelId]);

  useEffect(() => {
    setLayers((prev) => {
      let changed = false;
      const next = prev.map((l) => {
        const clamped = clampRanges(l.ranges, prompt.length);
        if (clamped.length !== l.ranges.length || clamped.some((r, i) => r[0] !== l.ranges[i][0] || r[1] !== l.ranges[i][1])) {
          changed = true;
          return { ...l, ranges: clamped };
        }
        return l;
      });
      return changed ? next : prev;
    });
  }, [prompt]);

  useEffect(() => {
    if (!modelOpen) return;
    const onDocClick = (e) => {
      if (modelFieldRef.current && !modelFieldRef.current.contains(e.target)) {
        setModelOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [modelOpen]);

  const tokensForCopy = useMemo(() => rows.map((r) => r.token).join(" "), [rows]);

  const setStatusKind = (text, kind = "idle") => setStatus({ text, kind });

  const loadTokenizer = useCallback(
    async (id) => {
      if (cacheRef.current.has(id)) return cacheRef.current.get(id);
      setStatusKind(`Loading ${id} tokenizer...`, "loading");

      const headers = hfToken ? { Authorization: `Bearer ${hfToken}` } : {};
      const base = `https://huggingface.co/${id}/resolve/main`;
      const tokenizerJson = await fetch(`${base}/tokenizer.json`, { headers }).then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch tokenizer.json (${r.status})`);
        return r.json();
      });
      const tokenizerConfig = await fetch(`${base}/tokenizer_config.json`, { headers }).then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch tokenizer_config.json (${r.status})`);
        return r.json();
      });

      const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
      const vocabSize = tokenizerJson.model?.vocab ? Object.keys(tokenizerJson.model.vocab).length : tokenizerJson.vocab?.length;
      const specialsSet = extractSpecialIds(tokenizerConfig);
      const cached = { tokenizer, vocabSize: vocabSize ?? "—", specialsSet };
      cacheRef.current.set(id, cached);
      return cached;
    },
    [hfToken]
  );

  const tokenize = useCallback(async () => {
    if (!prompt.trim()) {
      setStatusKind("Please add a prompt to tokenize.", "error");
      return;
    }

    const effectiveModelId = modelId.trim();

    if (!effectiveModelId) {
      setStatusKind("Please enter a Hugging Face model ID.", "error");
      return;
    }

    try {
      setStatusKind("Tokenizing...", "loading");
      const { tokenizer, vocabSize, specialsSet } = await loadTokenizer(effectiveModelId);
      const encoding = tokenizer.encode(prompt);

      const tokens = encoding.tokens;
      const ids = encoding.ids;

      const newRows = tokens.map((tok, idx) => ({
        token: tok,
        id: ids[idx],
        offset: ["—", "—"],
        special: specialsSet?.has(ids[idx]) || false,
      }));
      const specialCount = specialsSet ? ids.filter((id) => specialsSet.has(id)).length : 0;

      const offsets = [];
      let prevLen = 0;
      for (let i = 0; i < ids.length; i++) {
        let partialLen = prevLen;
        try {
          partialLen = tokenizer.decode(ids.slice(0, i + 1), { skip_special_tokens: false }).length;
        } catch {}
        offsets.push([prevLen, partialLen]);
        prevLen = partialLen;
      }
      try {
        const fullDecoded = tokenizer.decode(ids, { skip_special_tokens: false });
        if (fullDecoded.length > prompt.length) {
          const trimmed = fullDecoded.trimStart();
          if (trimmed === prompt || prompt.startsWith(trimmed.slice(0, Math.min(16, trimmed.length)))) {
            const shift = -(fullDecoded.length - trimmed.length);
            for (let i = 0; i < offsets.length; i++) {
              offsets[i] = [Math.max(0, offsets[i][0] + shift), Math.max(0, offsets[i][1] + shift)];
            }
          }
        }
      } catch {}

      setRows(newRows);
      setTokenOffsets(offsets);
      setSummary({
        vocab: vocabSize ?? "—",
        tokens: tokens.length,
        specials: specialCount,
      });
      setLastTokenizedModel(effectiveModelId);
      setStatusKind(`Tokenized with ${effectiveModelId}`, "idle");
    } catch (err) {
      console.error(err);
      setStatusKind(err.message || "Failed to tokenize", "error");
    }
  }, [loadTokenizer, modelId, prompt]);

  useEffect(() => {
    tokenizeRef.current = tokenize;
  }, [tokenize]);

  const clearAll = () => {
    setPrompt("");
    setRows([]);
    setTokenOffsets([]);
    setSummary({ vocab: "—", tokens: "—", specials: "—" });
    setStatusKind("Idle");
    setLayers((prev) => prev.map((l) => ({ ...l, ranges: [] })));
  };

  const addLayer = () => {
    setLayers((prev) => {
      const id = layerIdRef.current++;
      const color = LAYER_COLORS[prev.length % LAYER_COLORS.length];
      const next = [...prev, { id, name: `Layer ${prev.length + 1}`, color, ranges: [] }];
      setActiveLayerId(id);
      setActiveMode("edit");
      return next;
    });
  };

  const removeLayer = (id) => {
    setLayers((prev) => prev.filter((l) => l.id !== id));
    setActiveLayerId((cur) => (cur === id ? null : cur));
  };

  const renameLayer = (id, name) => {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, name } : l)));
  };

  const applyHighlightToRange = (range) => {
    if (!range || range[1] <= range[0]) return;
    if (activeLayerId == null) return;
    const snapped = snapRangeToTokens(range, tokenOffsets, prompt.length);
    if (!snapped) return;
    if (activeMode === "erase") {
      setLayers((prev) =>
        prev.map((l) => (l.id === activeLayerId ? { ...l, ranges: subtractRange(l.ranges, snapped) } : l))
      );
      return;
    }
    setLayers((prev) =>
      prev.map((l) => {
        if (l.id === activeLayerId) {
          return { ...l, ranges: normalizeRanges([...l.ranges, snapped]) };
        }
        return { ...l, ranges: subtractRange(l.ranges, snapped) };
      })
    );
  };

  const handlePromptSelection = (e) => {
    if (activeLayerId == null) return;
    const { selectionStart, selectionEnd } = e.target;
    if (selectionEnd > selectionStart) {
      applyHighlightToRange([selectionStart, selectionEnd]);
    }
  };

  const highlightSegments = useMemo(() => buildHighlightSegments(prompt, layers), [prompt, layers]);

  const tokenLayerMap = useMemo(() => {
    const map = new Array(tokenOffsets.length).fill(null);
    for (let i = 0; i < tokenOffsets.length; i++) {
      const [ts, te] = tokenOffsets[i];
      if (te > ts) {
        const mid = (ts + te) / 2;
        for (const l of layers) {
          if (l.ranges.some(([rs, re]) => mid >= rs && mid < re)) {
            map[i] = l.id;
            break;
          }
        }
      } else {
        for (const l of layers) {
          if (l.ranges.some(([rs, re]) => ts >= rs && ts <= re)) {
            map[i] = l.id;
            break;
          }
        }
      }
    }
    return map;
  }, [layers, tokenOffsets]);

  const layerColorById = useMemo(() => {
    const m = new Map();
    for (const l of layers) m.set(l.id, l.color);
    return m;
  }, [layers]);

  const layerTokenCounts = useMemo(() => {
    const counts = {};
    for (const l of layers) counts[l.id] = 0;
    for (const layerId of tokenLayerMap) {
      if (layerId != null) counts[layerId] = (counts[layerId] || 0) + 1;
    }
    return counts;
  }, [layers, tokenLayerMap]);

  const totalAssignedTokens = useMemo(
    () => Object.values(layerTokenCounts).reduce((a, b) => a + b, 0),
    [layerTokenCounts]
  );

  const promptTokenCount = typeof summary.tokens === "number" ? summary.tokens : 0;
  const ratioDenominator = ratioMode === "prompt" ? promptTokenCount : maxContext;

  const copyTokens = async () => {
    if (!tokensForCopy) return;
    try {
      await navigator.clipboard.writeText(tokensForCopy);
      setStatusKind("Tokens copied to clipboard.");
    } catch {
      setStatusKind("Clipboard unavailable.", "error");
    }
  };

  return (
    <>
      <header className="site-header">
        <div className="site-header__brand">
          <h1 className="site-header__title">Prompt Token Studio</h1>
          <span className="site-header__tagline">Plan a prompt against the context window</span>
        </div>
        <span className="site-header__poweredby">Powered by Reactor</span>
      </header>

      <div className="app-shell">
        <section className="hero">
          <p className="hero__lead">
            Slice your prompt into labeled layers and watch each one eat into the model&apos;s context window — built
            on the Reactor developer platform.
          </p>
        </section>

        <main className="grid">
        <section className="panel">
          <div className="controls-head">
            <div className="combobox combobox--inline" ref={modelFieldRef}>
              <input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                onFocus={() => setModelOpen(true)}
                type="text"
                placeholder="Hugging Face model"
                spellCheck={false}
              />
              <button
                type="button"
                className="combobox__arrow"
                aria-label="Show model presets"
                aria-expanded={modelOpen}
                onClick={() => setModelOpen((v) => !v)}
              >
                <svg viewBox="0 0 12 8" width="12" height="8" aria-hidden="true">
                  <path d="M1 1l5 5 5-5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              {modelOpen && (
                <ul className="combobox__list" role="listbox">
                  {models.map((m) => (
                    <li key={m.id}>
                      <button
                        type="button"
                        className="combobox__option"
                        onClick={() => {
                          setModelId(m.id);
                          setModelOpen(false);
                        }}
                      >
                        <span className="combobox__option-label">{m.label}</span>
                        <span className="combobox__option-id">{m.id}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <button
              type="button"
              className={`chip-toggle ${tokenOpen ? "chip-toggle--on" : ""}`}
              aria-expanded={tokenOpen}
              onClick={() => setTokenOpen((v) => !v)}
            >
              HF token
            </button>
            <button
              type="button"
              className={`chip-toggle ${maxOpen ? "chip-toggle--on" : ""}`}
              aria-expanded={maxOpen}
              onClick={() => setMaxOpen((v) => !v)}
            >
              Max: {maxContext || "—"}
            </button>
            <span
              className={`pill pill--inline ${
                status.kind === "loading" ? "pill--loading" : status.kind === "error" ? "pill--error" : ""
              }`}
            >
              {status.text}
            </span>
            <button type="button" className="btn btn--tiny" onClick={clearAll}>
              Clear
            </button>
          </div>

          {(tokenOpen || maxOpen) && (
            <div className="controls-expanded">
              {tokenOpen && (
                <input
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  type="password"
                  placeholder="hf_xxx (kept in memory)"
                />
              )}
              {maxOpen && (
                <input
                  value={maxContext}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    setMaxContext(Number.isFinite(v) && v > 0 ? v : 0);
                  }}
                  type="number"
                  min={1}
                  placeholder="Max context length"
                />
              )}
            </div>
          )}

          <div className="controls">
            <div className="field">
              <div className="prompt-editor">
                <div className="prompt-editor__backdrop" aria-hidden="true">
                  {highlightSegments.map((seg, i) => (
                    <span
                      key={i}
                      style={seg.color ? { background: seg.color + "55", borderRadius: 3 } : undefined}
                    >
                      {seg.text}
                    </span>
                  ))}
                  {prompt.endsWith("\n") ? "\n" : ""}
                </div>
                <textarea
                  ref={promptRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onMouseDown={() => {
                    const handler = () => {
                      document.removeEventListener("mouseup", handler);
                      const ta = promptRef.current;
                      if (!ta) return;
                      const { selectionStart, selectionEnd } = ta;
                      if (selectionEnd > selectionStart) {
                        applyHighlightToRange([selectionStart, selectionEnd]);
                      }
                    };
                    document.addEventListener("mouseup", handler);
                  }}
                  onKeyUp={handlePromptSelection}
                  rows={3}
                  spellCheck={false}
                  placeholder="Type something interesting to see how it gets split..."
                />
              </div>
            </div>

            <div className="field stream-section">
              <div className="stream-section__head">
                <button
                  type="button"
                  className="field__toggle"
                  aria-expanded={streamOpen}
                  onClick={() => setStreamOpen((v) => !v)}
                >
                  <span className={`field__caret ${streamOpen ? "field__caret--open" : ""}`}>▸</span>
                  Token breakdown
                  {rows.length > 0 && <span className="muted"> ({rows.length})</span>}
                </button>
                {streamOpen && rows.length > 0 && (
                  <button className="btn btn--tiny" onClick={copyTokens}>Copy tokens</button>
                )}
              </div>
              {streamOpen && (
                <div className="visualization">
                  {rows.length === 0 ? (
                    <p className="placeholder">Type a prompt to see color-coded tokens.</p>
                  ) : (
                    rows.map((row, idx) => {
                      const layerId = tokenLayerMap[idx];
                      const layerColor = layerId != null ? layerColorById.get(layerId) : null;
                      return (
                        <div key={idx} className="segment">
                          <span
                            className={`segment__token ${row.special ? "segment__token--special" : ""} ${layerColor ? "segment__token--layered" : ""}`}
                            style={layerColor ? { background: layerColor + "44" } : undefined}
                          >
                            {row.token}
                          </span>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          </div>

        </section>

        <section className="panel panel--tall">
          <div className="layers-head">
            <div className="tokens-produced">
              <span className="label">Tokens produced</span>
              <span className="tokens-produced__value">{summary.tokens}</span>
            </div>
            <div className="layers-head__bar">
              <div className="budget__bar" aria-hidden="true">
                {layers.map((l) => {
                  const count = layerTokenCounts[l.id] ?? 0;
                  const width = ratioDenominator > 0 ? Math.min(100, (count / ratioDenominator) * 100) : 0;
                  if (width === 0) return null;
                  return (
                    <span
                      key={l.id}
                      className="budget__bar-seg"
                      style={{ width: `${width}%`, background: l.color }}
                      title={`${l.name}: ${count} tokens`}
                    />
                  );
                })}
              </div>
              <span className="budget__total">
                {totalAssignedTokens} / {ratioDenominator || "—"} tokens {ratioMode === "prompt" ? "in prompt" : "assigned"}
              </span>
            </div>
            <div
              className="ratio-toggle"
              role="group"
              aria-label="Ratio denominator"
            >
              <button
                type="button"
                className={`ratio-toggle__option ${ratioMode === "prompt" ? "ratio-toggle__option--on" : ""}`}
                onClick={() => setRatioMode("prompt")}
                title="Show each layer as a share of the total prompt"
              >
                Prompt %
              </button>
              <button
                type="button"
                className={`ratio-toggle__option ${ratioMode === "context" ? "ratio-toggle__option--on" : ""}`}
                onClick={() => setRatioMode("context")}
                title="Show each layer as a share of the model's max context"
              >
                Context %
              </button>
            </div>
          </div>

          <div className="layer-toolbar">
            <span className="label">Layers</span>
            <div className="layer-toolbar__actions">
              <button type="button" className="btn btn--tiny" onClick={addLayer}>+ Add layer</button>
            </div>
          </div>

          {layers.length === 0 ? (
            <p className="layer-hint">No layers yet. Click <strong>+ Add layer</strong>, then highlight prompt regions while the layer is active.</p>
          ) : (
            <ul className="layer-list">
              {layers.map((l) => {
                const count = layerTokenCounts[l.id] ?? 0;
                const pct = ratioDenominator > 0 ? (count / ratioDenominator) * 100 : 0;
                const isActive = activeLayerId === l.id;
                const toggleActive = () => {
                  setActiveLayerId((v) => {
                    if (v === l.id) return null;
                    return l.id;
                  });
                  setActiveMode("edit");
                };
                return (
                  <li
                    key={l.id}
                    className={`layer-card ${isActive ? "layer-card--active" : ""}`}
                    onClick={(e) => {
                      if (e.target.closest(".layer-card__name, .layer-card__delete")) return;
                      toggleActive();
                    }}
                  >
                    <div
                      className={`mode-toggle mode-toggle--${isActive ? activeMode : "edit"} ${isActive ? "" : "mode-toggle--inactive"}`}
                      style={{ "--layer-color": l.color }}
                      onClick={(e) => e.stopPropagation()}
                      role="group"
                      aria-label="Layer mode"
                    >
                      <span className="mode-toggle__knob" aria-hidden="true" />
                      <button
                        type="button"
                        className={`mode-toggle__option ${isActive && activeMode === "edit" ? "mode-toggle__option--on" : ""}`}
                        title="Edit — add selection to this layer"
                        onClick={() => { setActiveLayerId(l.id); setActiveMode("edit"); }}
                      >
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <path d="m9 11-6 6v3h9l3-3" />
                          <path d="m22 12-4.6 4.6a2 2 0 0 1-2.8 0l-5.2-5.2a2 2 0 0 1 0-2.8L14 4l8 8z" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        className={`mode-toggle__option ${isActive && activeMode === "erase" ? "mode-toggle__option--on" : ""}`}
                        title="Erase — remove selection from this layer"
                        onClick={() => { setActiveLayerId(l.id); setActiveMode("erase"); }}
                      >
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                          <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21" />
                          <path d="M22 21H7" />
                          <path d="m5 11 9 9" />
                        </svg>
                      </button>
                    </div>
                    <svg className="layer-card__ring" viewBox="0 0 36 36" width="36" height="36" aria-hidden="true">
                      <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="3.5" />
                      <circle
                        cx="18"
                        cy="18"
                        r="14"
                        fill="none"
                        stroke={l.color}
                        strokeWidth="3.5"
                        strokeLinecap="round"
                        strokeDasharray={2 * Math.PI * 14}
                        strokeDashoffset={2 * Math.PI * 14 * (1 - Math.min(1, pct / 100))}
                        transform="rotate(-90 18 18)"
                      />
                      <text x="18" y="20" textAnchor="middle" className="layer-card__ring-label">
                        {ratioDenominator > 0 ? `${Math.round(pct)}%` : "—"}
                      </text>
                    </svg>
                    <input
                      className="layer-card__name"
                      value={l.name}
                      onChange={(e) => renameLayer(l.id, e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <span className="layer-card__count">
                      {count} <span className="muted">/ {ratioDenominator || "—"}</span>
                    </span>
                    <button
                      type="button"
                      className="layer-card__delete"
                      aria-label="Delete layer"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeLayer(l.id);
                      }}
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          )}

        </section>
      </main>

        <footer className="footer">
          <div>
            Tokenization runs locally via <code>@huggingface/tokenizers</code>. Models are fetched from the Hugging Face Hub; nothing leaves the browser.
          </div>
          <div className="footer__links">
            <a
              className="footer__link"
              href="https://docs.reactor.inc/overview"
              target="_blank"
              rel="noreferrer"
            >
              docs.reactor.inc
            </a>
          </div>
        </footer>
      </div>
    </>
  );
}
