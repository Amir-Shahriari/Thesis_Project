# Qwarm Reach Telemetry — Design System

Tailored for the ICDM 2026 demo-track booth build of `demo_app/static/index.html`.

**Product type:** real-time scientific / ML telemetry dashboard
**Context:** laptop + projector at a conference booth; read from ~3 m; ~2 min attention span
**Style:** dark, high-contrast, control-room telemetry, motion-forward, minimal chrome

---

## 0. Hard constraints these tokens obey

| # | Constraint | Source | Consequence |
|---|---|---|---|
| C1 | Zero outbound network at runtime | `tests/test_offline.py` | No webfonts, no CDN, no GSAP. Local font stacks + inline CSS/JS only. |
| C2 | No fabricated numbers anywhere in the UI | `verify_demo_claims.py`, curation comment in `index.html` | Counters may only animate toward values the server actually returned. |
| C3 | Colour-vision-deficiency safe | existing Okabe-Ito comment block | Every agent distinction carries a redundant non-colour channel. |
| C4 | Wire format has **no per-step cost and no per-step latency**; `edges` are `[u,v]` with no weights | `server.py:265-274`, `server.py:416-428` | Per-step running cost is not derivable client-side. See §6. |

---

## 1. Colour tokens

Base surfaces are pushed deeper than the incumbent (`#0a0e14` → `#05080E`) so projector black
crushes less and the agent hues gain headroom.

### Surfaces & chrome

| Token | Value | Role |
|---|---|---|
| `--bg` | `#05080E` | page ground |
| `--surface` | `#0B111A` | panel / card |
| `--surface-2` | `#121A26` | inputs, raised |
| `--surface-3` | `#1A2534` | segmented controls, table head |
| `--line` | `#223044` | borders |
| `--line-soft` | `#16202E` | internal dividers |

### Type

| Token | Value | Contrast on `--surface` | Grade |
|---|---|---|---|
| `--text` | `#F2F7FF` | 17.60 | AAA |
| `--text-dim` | `#A3B4CA` | 8.96 | AAA |
| `--text-faint` | `#6B7F99` | 4.62 | AA |

### Agent identity — the emotional hero

| Token | Value | Contrast | Deuter. sep. vs warm | Role |
|---|---|---|---|---|
| `--warm` | `#00E5A0` | 11.46 AAA | — | demonstration-assisted. Confident cyan-green. |
| `--cold` | `#FF7A1A` | 7.26 AAA | 0.781 | demonstration-free. Alarm amber. |
| `--cold-alarm` | `#FF4D4D` | 5.79 AA | — | the dead-end moment only. |
| `--dijk` | `#6E86A6` | 5.07 AA | 0.186 | muted neutral reference. Recedes. |

**Validated, not eyeballed.** Against the incumbent Okabe-Ito triad:

- warm/cold separation under deuteranopia improves **0.549 → 0.781**, protanopia **0.429 → 0.589**.
- warm/dijk improves **0.163 → 0.186** (deuter) and **0.197 → 0.240** (protan).
- Dijkstra's contrast rises **3.65 (below AA) → 5.07 (AA)**.

So the brief's "brighter, more emotional" direction and CVD safety are not in tension here —
brightening *widened* every hue separation. Two caveats that drive design decisions below:

> **Caveat A — warm vs Dijkstra is carried by luminance and line style, not hue.**
> Even at the improved 0.186, that pair is weakly separated under red-green CVD. It is legible
> because luminance differs by 0.355 (warm `L=0.586`, dijk `L=0.298`) *and* Dijkstra is dotted
> against warm's solid. Both channels are load-bearing — neither may be dropped.

> **Caveat B — the amber→red dead-end transition is nearly invisible under CVD** (separation
> 0.058 deuter, 0.029 tritan). The dead-end moment must therefore be signalled by the halo, the
> `×` glyph, and an explicit text chip. Colour is decoration there, never the message.

### Status & graph

| Token | Value | Role |
|---|---|---|
| `--ok` | `#00E5A0` | reached — deliberately identical to `--warm` |
| `--fail` | `#FF4D4D` | failed |
| `--warn` | `#FFB020` | OOD / caution notes |
| `--accent` | `#3D8BFF` | UI chrome only; never an agent colour |
| `--n-start` | `#22D3EE` | start node |
| `--n-goal` | `#F0ABFC` | goal node |
| `--n-active` | `#46586F` | live node |
| `--n-off` | `#1B2436` | deactivated by perturbation |
| `--edge` | `#16202F` | graph edge |

### Redundant encoding contract (C3)

| Agent | Colour | Line | Head shape |
|---|---|---|---|
| warm | cyan-green | solid | circle |
| cold | amber | dashed `[11,7]` | square |
| Dijkstra | muted slate | dotted `[2,7]` | none |

---

## 2. Type scale

Local stacks only (C1). Roles borrowed from the DB's "Dashboard Data" pairing (mono for data,
sans for labels); the incumbent already declares compatible stacks, so these are unchanged.

```
--font-sans: "Inter", ui-sans-serif, system-ui, "Segoe UI Variable Text", "Segoe UI", Roboto, sans-serif;
--font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, Consolas, monospace;
```

| Token | Size | Use |
|---|---|---|
| `--fs-display` | `clamp(44px, 5vw, 76px)` | REACHED / FAILED verdict. Mono, 700, `-0.03em`. |
| `--fs-metric` | `clamp(28px, 2.8vw, 42px)` | gap, latency, steps. Mono, 650. |
| `--fs-title` | `17px` | page title |
| `--fs-body` | `14px` | prose |
| `--fs-readout` | `13px` | mono panel footers |
| `--fs-label` | `10px` | uppercase, `letter-spacing: .1em` |

All numeric readouts use `font-variant-numeric: tabular-nums` so ticking counters do not reflow.

---

## 3. Spacing — dense (dial 8/10)

`--sp-1: 4px` · `--sp-2: 8px` · `--sp-3: 12px` · `--sp-4: 16px` · `--sp-5: 24px` · `--sp-6: 32px` · `--sp-7: 48px`

Chrome stays dense; the hero metric row alone gets `--sp-6`/`--sp-7` breathing room, because
that is the one region that must carry across a room.

Radii: `--r-sm: 6px` · `--r-md: 10px` · `--r-lg: 14px`

---

## 4. Motion tokens

```
--dur-fast:   140ms   /* chrome: hover, focus            */
--dur-base:   200ms   /* state changes                   */
--dur-slow:   320ms   /* panel ignition                  */
--dur-ignite: 460ms   /* verdict reveal                  */
--ease-out:   cubic-bezier(.16, 1, .3, 1)
--ease-inout: cubic-bezier(.65, 0, .35, 1)
--ease-alarm: cubic-bezier(.36, .07, .19, .97)
```

Durations sit in the DB's 150–300 ms band for UI state; only the two narrative moments
(verdict, alarm) exceed it, and both are one-shot.

### Motion inventory

The DB's guidance is "animate 1–2 key elements per view maximum". These five are scoped so no
more than two run concurrently in any one region.

| ID | Motion | Trigger | Spec |
|---|---|---|---|
| **M1** | Progressive path draw + fading trail | scrubber / play | Existing rAF loop. Traversed path at α 0.45; last 6 segments overdrawn at α 1.0 with 8px glow → comet tail. |
| **M2** | Counter tick | run completion | 460 ms `--ease-out` count-up. **Lands on the exact server value** (C2). |
| **M3** | Status ignition | verdict resolves | Badge scale `.92→1`, panel border colour + shadow bloom over `--dur-slow`. |
| **M4** | Dead-end alarm | the exact step cold gets stuck | One-shot 2-ring expanding pulse + `×` + `DEAD END @ STEP n` chip. |
| **M5** | Chrome micro | hover / focus | 140 ms colour + border only. Never width/height. |

`@media (prefers-reduced-motion: reduce)` disables M1–M4 (paths and counters snap to final
state) and keeps M5 at 0.01 ms. No information is motion-only.

---

## 5. Component specs

**Hero verdict cards (warm | cold).** Two oversized cards, `--fs-display`, colour-coded via
`--warm`/`--fail`. The single loudest thing on screen. Sub-line carries the honest qualifier
(`dead end, no unvisited move` vs `step budget exhausted`) at `--fs-readout`.

**Secondary metric row.** Optimality gap, latency, steps — `--fs-metric`, mono, left rule in
the owning agent's colour.

**Three canvases.** Unchanged geometry and transform. Panel border ignites (M3) on the winner.

**Scrubber.** Taller track (10px), thumb 20px, plus a **dead-end tick mark** rendered on the
rail at cold's stuck step so the failure is locatable before you scrub to it.

**Legend.** Must show line style *and* head shape per agent, not colour swatches alone (C3).

---

## 6. Open decision — per-step counters (C4)

Goal 3 asks for "live-ticking metric counters". The wire format supports this only partially:

| Metric | Per-step honest? | Treatment |
|---|---|---|
| steps | **Yes** — equals the scrubber index | Ticks live during playback. |
| path progress | **Yes** — derived from path length | Ticks live. |
| cost | **No** — aggregate only, edges carry no weights | Counts up once on completion (M2), labelled as a final aggregate. |
| optimality gap | **No** — derived from aggregate cost | Same. |
| latency | **No** — `mean_latency_ms` only | Same, labelled "mean, per decision". |

Default plan: the honest split above. A per-step cost readout would require adding a
cumulative-cost array to the `/reach` response — a small backend change, offered but not
assumed, since the brief said not to disturb the data flow.
