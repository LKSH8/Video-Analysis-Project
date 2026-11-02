<USER_REQUEST>
i want to make ot less vibecoded this is the proimpt do the changes 
# DESIGN.md — Forensic Video Analysis Interface

Read this before writing or changing any UI. Follow it literally. When a rule here conflicts with a default habit, the rule wins.

## 1. Concept

The interface is an **evidence examination workstation**, not a SaaS dashboard. Think of a forensic lab terminal, an instrument readout, a case file. The user is an analyst reviewing footage, not a customer being sold a product.

Design principles:

1. **Instrument, not brochure.** Every element exists to display data or take an action. No decoration.
2. **Evidence is the hero.** Video frames and detections get the most space and the highest contrast. Chrome recedes.
3. **Provenance everywhere.** Timestamps, frame numbers, camera IDs, track IDs and confidence values are always visible next to what they describe.
4. **Dense but ordered.** Tight spacing, strict alignment, a visible grid. Density comes from structure, not from shrinking text.
5. **Quiet by default.** Colour is reserved for meaning. Most of the screen is neutral.

## 2. Banned patterns (the "vibecoded" tells)

Do not use any of the following, even if they seem like a good idea:

- Gradients of any kind: backgrounds, buttons, text, borders
- Glassmorphism, backdrop blur, frosted panels
- Glow, neon, outer colour shadows, pulsing halos
- Purple, indigo, violet, or blue-to-purple anything
- Cyan-on-black "cyberpunk" or "hacker" green-on-black styling
- Emoji as icons or as section decoration
- Sparkle / magic-wand / robot icons, "AI-powered" badges, "✨"
- Large rounded corners (`rounded-xl`, `rounded-2xl`, pill buttons, circular avatars for non-people)
- Cards-in-cards, and a card around every single thing
- Hero sections, taglines, marketing copy, centred headline + subtitle
- Three-column feature grids with icon + title + sentence
- Big drop shadows to create "depth". Elevation is not used here.
- Hover lift / scale animations, bouncing, shimmer skeletons, typewriter effects
- Gradient progress bars, animated gradient borders
- Inter / Poppins / Space Grotesk as the display font
- Friendly empty-state illustrations and chatty microcopy ("Oops!", "Let's get started!")
- Centre-aligned body content; everything is left-aligned to the grid
- Identical padding on every element (`p-6` everywhere)

## 3. Colour

Flat, desaturated, warm-neutral graphite. Accent colours are functional signals, never decoration.

### Tokens

```css
:root {
  /* Surfaces (dark workstation) */
  --bg-base:      #0E0F10;   /* app background */
  --bg-panel:     #151617;   /* panels, sidebars */
  --bg-inset:     #0A0B0C;   /* video wells, data wells, inputs */
  --bg-raised:    #1C1D1F;   /* table header, active row */

  /* Lines */
  --line-hair:    #26282A;   /* default 1px divider */
  --line-strong:  #3A3D40;   /* panel outlines, focus-adjacent */

  /* Text */
  --text-primary:   #D8DADC;
  --text-secondary: #8C9094;
  --text-tertiary:  #5C6064;
  --text-inverse:   #0E0F10;

  /* Signal colours (functional only) */
  --sig-amber:    #D9A441;   /* timestamps, selection, primary action */
  --sig-red:      #C8453B;   /* alerts, tamper, low confidence, REC state */
  --sig-green:    #5E9C76;   /* verified, integrity OK, high confidence */
  --sig-steel:    #6E8CA3;   /* neutral info, links, secondary markers */

  /* Detection overlay classes (bounding boxes) */
  --det-person:   #D9A441;
  --det-vehicle:  #6E8CA3;
  --det-object:   #5E9C76;
  --det-flagged:  #C8453B;
}
```

### Rules

- Max **one** signal colour dominant on any screen region. Amber is the default.
- Text on `--bg-*` must meet WCAG AA (4.5:1). `--text-tertiary` is for non-essential labels only.
- Red means something is wrong or flagged. Never use it for decoration or for a "delete" button unless it is truly destructive.
- Status is never colour-only: pair with a text label or a glyph (`OK`, `FLAG`, `LOW`).
- Optional light mode ("print / report view"): paper `#F2F1EE`, ink `#1A1B1C`, same signal hues darkened ~15%. Used only for exported reports.

## 4. Typography

Two families, both technical and neutral.

- **UI text:** `IBM Plex Sans`, fallback `system-ui, sans-serif`
- **Data, IDs, timestamps, coordinates, hashes, logs:** `IBM Plex Mono`, fallback `ui-monospace, monospace`

```css
--font-ui:   "IBM Plex Sans", system-ui, sans-serif;
--font-data: "IBM Plex Mono", ui-monospace, monospace;
```

| Role | Font | Size / line | Weight | Case |
|---|---|---|---|---|
| Page title | UI | 20 / 28 | 600 | Sentence |
| Panel heading | UI | 11 / 16 | 600 | UPPERCASE, +0.08em tracking |
| Body | UI | 13 / 20 | 400 | Sentence |
| Table cell | UI | 12.5 / 18 | 400 | Sentence |
| Data value | Mono | 12.5 / 18 | 400 | As-is |
| Timestamp / frame no. | Mono | 12 / 16 | 500 | `tabular-nums` |
| Micro label | UI | 10.5 / 14 | 500 | UPPERCASE, +0.1em tracking |

Rules:

- Base size is **13px**, not 16px. This is a dense tool.
- All numerics use `font-variant-numeric: tabular-nums` so columns align.
- Timestamps always render as `YYYY-MM-DD HH:MM:SS.mmm` in mono. Frame numbers as `F000482`. Track IDs as `TRK-0031`. Camera IDs as `CAM-03`.
- Never use more than two weights per screen (400 and 600).
- No gradient text, no text shadows, no italics for emphasis.

## 5. Layout and spacing

- **Spacing scale (px):** 2, 4, 8, 12, 16, 24, 32. Nothing else. Default gap 8 or 12, not 24.
- **Grid:** 12-column, 16px gutters, with a visible 1px hairline between major regions rather than whitespace alone.
- **Shell:** fixed left rail (56px icon rail or 240px tree), top status bar (32px), main workspace, optional right inspector (320px), bottom timeline (96–140px).
- **Panels** are rectangles with a 1px `--line-hair` border and no shadow. Panel header is a 28px strip with a micro label on the left and tools on the right, separated by a bottom hairline.
- **Radius:** `0` for panels and tables, `2px` for buttons, inputs and tags. Never above `4px`.
- Content is left-aligned. Numbers are right-aligned in tables. Nothing is centred except inside the video well when letterboxing.
- Vary density deliberately: the video well is generous, the metadata tables are tight. Do not apply uniform padding.

## 6. Components

### Top status bar
Single 32px row, mono text, `--bg-panel`, bottom hairline. Left: case ID and source file. Centre: none. Right: system clock (UTC), processing state, integrity state. Example: `CASE 2026-0142 · SRC cam03_lobby.mp4 · SHA-256 9F3A…C1D2 · UTC 14:07:22`.

### Video well
- Background `--bg-inset`, 1px `--line-strong` border, no radius.
- Burned-in style overlay in mono, amber, top-left: `CAM-03  2026-10-14 14:07:22.480  F000482`. Bottom-right: resolution and fps.
- Corner registration ticks (L-shaped, 10px, 1px, `--text-secondary`) at the four corners instead of a rounded frame.
- Optional centre crosshair only when the user enables it.

### Detection boxes
- 1px solid stroke (1.5px max), no fill, no glow, no rounded corners.
- Label tab sits on the top-left edge: mono, 10.5px, class colour background, `--text-inverse` text. Format: `PERSON 0.91 · TRK-0031`.
- Flagged detections switch to `--det-flagged` and add a `FLAG` suffix.
- Trails: 1px polyline, 50% opacity, fading by dropping alpha steps, not gradients.

### Timeline
- Bottom strip with a hairline time ruler (major ticks each minute, minor each 10s), mono labels.
- Event markers are 2px wide vertical bars coloured by class. The playhead is a 1px amber line with a small square handle.
- Density histogram uses flat 1px-wide bars in `--text-tertiary`; selection range is `--sig-amber` at 12% alpha.

### Tables (event log / query results)
- No zebra stripes, no card rows. 1px hairline between rows, header on `--bg-raised`.
- Columns: `#`, `TIME`, `CAM`, `TRK`, `CLASS`, `CONF`, `MATCH`, `ACTION`.
- Row height 28px. Selected row: left 2px amber bar and `--bg-raised` fill.
- Confidence shown as a number (`0.91`) with a 40px flat bar behind it, no gradient.
- Sortable header shows a small `▲/▼` glyph in text, not an icon-font chevron.

### Search / query input
- Full-width inset field, 32px tall, mono text, 2px radius, `--line-strong` border, amber 1px border on focus (no ring, no glow).
- Prefix label `QUERY` in micro style. Example placeholder: `person in dark jacket near entrance, 14:00–15:00`.
- Results show a **match score** and a **frame thumbnail with timestamp**, never a chat bubble.

### Buttons
- Primary: solid `--sig-amber`, `--text-inverse` text, 2px radius, 28px tall, 600 weight, uppercase micro label style (`RUN ANALYSIS`).
- Secondary: transparent, 1px `--line-strong` border, `--text-primary`.
- Tertiary: text only, `--text-secondary`, underline on hover.
- Destructive: outlined in `--sig-red`, never filled by default.
- Hover = background shifts one surface step lighter. No lift, no scale, no shadow.
- Disabled = `--text-tertiary` with no border change.

### Tags / status chips
- Rectangular, 2px radius, 1px border in the signal colour, transparent fill, mono 10.5px uppercase text. e.g. `VERIFIED`, `FLAGGED`, `PROCESSING`.
- No pastel filled pills.

### Inspector (right panel)
Key–value list: micro label above, mono value below, hairline between pairs. Sections: Object, Track, Frame, Source, Integrity. Include a copy affordance for hashes and IDs.

### Progress and loading
- Progress is a 2px flat line with the percentage and counts in mono (`FRAME 4,812 / 18,000 · 26.7%`).
- Loading states are a static text line such as `PROCESSING…` in amber. No spinners with gradients, no skeleton shimmer.

### Empty and error states
- Plain, factual, mono where it helps: `NO MATCHES FOR QUERY` / `0 EVENTS IN RANGE 14:00–15:00`. One neutral hint line at most. No illustrations, no exclamation marks.
- Errors state what failed and where: `DECODE ERROR · F001204 · CAM-03`.

## 7. Iconography

- Use a single outline set at 1.5px stroke, 16px size (Lucide or Phosphor Regular). Do not mix sets.
- Prefer text labels over icons for primary actions. Icons only for universal tools (play, pause, step, zoom, export, filter).
- No filled/duotone icons, no coloured icon backgrounds, no icons inside circles.
- Custom glyphs allowed and encouraged: crosshair, corner ticks, frame-step marks, ruler ticks.

## 8. Motion

- Default: none. State changes are instant.
- Allowed: 120ms linear opacity/background-colour transitions on hover and selection; playhead and video playback.
- Forbidden: easing curves with overshoot, entrance animations, parallax, hover transforms, count-up number animations.
- Respect `prefers-reduced-motion`.

## 9. Voice and microcopy

- Terse, declarative, technical. Labels are nouns. Status is a state word.
- Use forensic vocabulary: *Exhibit, Source, Frame, Track, Detection, Confidence, Match, Integrity, Export, Log*.
- No second-person chatty text, no "Let's", "Awesome", "Oops", "magic", "smart", "powered by AI".
- Show units and precision always: `0.91`, `25 fps`, `1920×1080`, `00:03:42.480`.
- Dates are ISO 8601. Times are 24-hour. State the timezone (UTC) in the status bar.

## 10. Detail touches that sell the theme

- Thin 1px registration ticks on panel corners for the video well and key insets only (not every panel).
- Case header block with case number, examiner field, and a chain-of-custody log (action, user, UTC time).
- Hash display for source files with a copy button and an `INTEGRITY OK / MISMATCH` state.
- Export produces a report with the same tokens in light mode, a header block, and numbered exhibits.
- Scale bar or pixel-coordinate readout in the inspector when hovering a frame (`X 0812  Y 0344`).
- Keyboard shortcut hints in mono next to actions (`SPACE`, `J K L`, `[` `]`). Analysts expect them.

## 11. Streamlit implementation notes

Streamlit's defaults are a strong vibecoded signal, so override them.

`.streamlit/config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#D9A441"
backgroundColor = "#0E0F10"
secondaryBackgroundColor = "#151617"
textColor = "#D8DADC"
font = "monospace"
```

Inject global CSS once with `st.markdown(..., unsafe_allow_html=True)`:

```css
html, body, [class*="css"] { font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 13px; }
code, pre, [data-testid="stMetricValue"], .stDataFrame { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
.block-container { padding: 1rem 1.25rem; max-width: 100%; }
button, input, textarea, [data-baseweb="select"] > div { border-radius: 2px !important; }
[data-testid="stSidebar"] { border-right: 1px solid #26282A; background: #151617; }
div[data-testid="stMetric"] { border: 1px solid #26282A; padding: 8px 12px; border-radius: 0; background: #151617; }
h1, h2, h3 { font-weight: 600; letter-spacing: 0; }
hr { border-color: #26282A; }
header[data-testid="stHeader"] { background: transparent; }
```

- Remove default page icons, emoji in `st.set_page_config` and in tab/expander titles.
- Replace `st.metric` delta arrows with plain mono text where possible.
- Render detection overlays with OpenCV: 1px boxes, mono label tabs, no anti-aliased glow.
- Use `st.dataframe` with column config for right-aligned numerics; avoid `st.balloons`, `st.snow`, and `st.success` banners with emoji.

## 12. Acceptance checklist

Before finishing any UI change, confirm:

- [ ] No gradients, blur, glow, shadows or emoji anywhere
- [ ] Radius is 0 or 2px everywhere
- [ ] Only one signal colour dominant per region, colour carries meaning
- [ ] All timestamps, IDs and numbers are mono and tabular
- [ ] Every detection or result shows time, camera and confidence
- [ ] Spacing uses the scale only; density varies by region
- [ ] Text is terse and technical; no marketing or chatty copy
- [ ] No hover lift, entrance animation or shimmer
- [ ] Looks like a lab instrument at a glance, not a landing page
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-02T02:13:17+05:30.

The user has uploaded 2 image(s):
- C:/Users/sehra/.gemini/antigravity/brain/17e1c31c-b472-4311-81b6-d7afd6164377/.user_uploaded/media_1793565778688_8d80a5a4.png
- C:/Users/sehra/.gemini/antigravity/brain/17e1c31c-b472-4311-81b6-d7afd6164377/.user_uploaded/media_1793565785189_5ca2ba5b.png
You can embed these images in an artifact if you need the USER to review them.
</ADDITIONAL_METADATA>