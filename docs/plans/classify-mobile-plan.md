# Classify page on phones: full-width image & no face labels

**Status:** Implemented 2026-10-18; label toggle added 2026-10-23 · **Created:** 2026-10-18 · **Owner:** WikiVisage maintainer

**Related:** `templates/classify.html` (classify UI CSS) · request from 2026-10-18 phone screenshot: *"the image does not go to the width of the screen"* and *"on phone screens, in both vertical and horizontal mode, I don't want these FACE 1 / FACE 2 annotations"*.

Revision 2 integrates two independent reviews (technical/CSS + UX/product). No blockers were found. This revision adds the worst-case measurements, the screenshot/orientation reconciliation, a verified edge-to-edge option, a sharpness decision, and small accuracy fixes.

**Revision 3 (2026-10-23, supersedes parts of §3):** the unconditional touch-screen `display: none` on `.face-label` became a **user toggle**. Touch devices still start with labels hidden, but a "Show/Hide labels" button (multi-face pages only) and an `H` hotkey flip them, and the choice is persisted in `localStorage` (`wikivisage-face-labels`). Only faces 1–9 get a visual tag now, matching the digit shortcuts. `tests/test_app.py` pins both the 9-label cap and the absence of the toggle on single-face pages. §3 below records the original change; §3.2 describes the toggle.

**Implementation:** the §3 block is live in `templates/classify.html` (appended after the mobile media query, before `</style>`), with the recommended decisions taken: D1a accept scroll · D2 hints hidden · D3 all touch devices · D4 container gutter · D5 1280 px source for now. The §9 harness, regenerated from the edited template, reproduced the §4 "After" numbers exactly; the desktop rows were unchanged.

---

## Scope

**In scope**

- `templates/classify.html`: one `@media (hover: none) and (pointer: coarse)` block (5 declarations) appended at the end of the page's existing `<style>` block, plus the Revision 3 label-toggle script (a few dozen lines, no dependency). Original change, CSS only:
  - the image spans the full content width on touch devices, in **both** orientations;
  - the `Face N` overlay labels and the keyboard hotkey-hints row are hidden on touch devices by default (Revision 3 adds a persisted show/hide toggle for the labels).
- Desktop (mouse / hover-capable devices): rendering and labels stay exactly as today.

**Out of scope / verified unaffected**

- JavaScript: `positionBoxes()` derives box positions from the image's displayed size (`img.clientWidth` / `offsetLeft`), so boxes follow the resized image with no JS change (verified; §4). Revision 3 adds only a small label-toggle script (§3.2).
- `app.py`, `worker.py`, `model-server/`, database, i18n, and all other templates (`image-wrapper` / `face-label` / `hotkey-hints` exist only in `classify.html`).
- Tests: Revision 2 deliberately added none (CSS-only). Revision 3's JS toggle is pinned by `test_classify_labels_only_number_first_nine_faces` and `test_classify_single_face_has_no_label_toggle` in `tests/test_app.py`.

**Open options** (decisions in §6): edge-to-edge bleed, sharper source (srcset), pinned controls, tablet scope.

---

## 1. Problem

- **Reported (screenshot), landscape phone:** the 1280×853 photo renders **276×184 CSS px** inside an 852 px-wide content column, ~29 % of the 956 px screen, with the overlapping `Face N` chips piled onto the postage-stamp image.
- **The same setup with a portrait-orientation photo is worse:** **123×184 px** (~13 % of the screen).
- **Portrait phone, portrait-orientation photo:** capped to **287×430 px** (65 % of the screen width), the "does not go to the width of the screen" case for upright use.
- **Labels:** rendered on every viewport whenever the image has more than one face; no rule hides them on phones.

## 2. Root cause (measured)

Two height caps in `templates/classify.html`, each correct for its original purpose, combine badly with phones:

1. **The "desktop short screens" rule** (lines 355–360):

   ```css
   @media (min-width: 601px) and (max-height: 900px) {
       .image-wrapper img { max-height: calc(100vh - 16rem); }
   }
   ```

   A phone in landscape is *short and wide*, exactly what this query describes: `width: 956 > 601` and `height: 440 < 900` both match, so the cap becomes `100vh − 16rem = 440 − 256 = 184 px`. The img is height-capped, so its width shrinks proportionally (276 px for the landscape photo, 123 px for a portrait photo). The wrapper shrink-wraps it, and `min-height: 300px` leaves the empty surface area below, exactly the screenshot.

2. **The mobile rule** (lines 362–412): `@media (max-width: 600px) { .image-wrapper img { max-height: 45vh } }`. On a 440×956 portrait phone the cap is 430 px, so a portrait-orientation photo renders **287×430 px** instead of filling the width.

The `Face N` labels (`.face-box .face-label`, line 79 ff.) have no phone-specific rule at all.

**No engine bug is involved**: plain media-query math; it reproduces in any browser. (An earlier "WebKit shrink-to-fit + 300 px replaced element" hypothesis was disproven by direct measurement: plain WebKit renders the image 844 px wide at a 1024 px viewport.)

### 2.1 Reproduction evidence

The delivered screenshot is a re-encoded 748×343 crop (not the original capture). **Decisive test (width share):** the photo spans only ~29 % of the page width (≈214 of ~720 slice px), while in every normal upright rendering this same photo fills 92–100 % of the width (measured: 404 px at a 440 px viewport; 844 px in desktop-mode). The frame is therefore the short-and-wide layout, and the capped height pins the mechanism exactly: `100vh − 16rem` at vh = 440. Converting via anchors inside the frame (content column, badge/button edges) gives a scale of ≈0.73–0.75 slice-px per CSS px → an iPhone-class landscape viewport of ≈950–1030 CSS px; the harness uses the nominal 956×440. Measured against the faithful WebKit replica (real CSS from both templates, the same Commons file as the chip text shows, real 1280 px thumb):

| Metric | User screenshot (CSS px) | WebKit replica @ 956×440 |
|---|---|---|
| Image width | ~270–276 | **276.1** |
| Image height (the capped dimension) | 184 | **184.0** |
| Wrapper width | ~284 | **284.1** |
| Wrapper height | ≥ 300 (cut off by crop) | **300** (`min-height`) |

The **capped height matches to the pixel** and the wrapper width to 0.04 %; the image width agrees within 1–2 % (the delivered file is a derivative and its exact thumb aspect differs slightly).

**Reconciling the screenshot with "vertical mode":** the numbers describe a **956×440 CSS viewport, the short-and-wide (landscape) layout**. If the screenshot was taken with the phone upright, some forced-wide rendering (page zoom, a webview, desktop mode) must have been in effect, although we could not reproduce the tiny image in any upright configuration; what the width share shows for certain is that the frame is not the normal upright layout. Either way: (a) this fix covers **both** orientations, and (b) Stage 2 closes the loop with a real-device check; if the tiny image still reproduces **upright** after the update, the plan re-opens with the *original full-resolution* screenshot as the first piece of evidence.

## 3. The fix

Append one block at the **end** of the `<style>` block in `templates/classify.html` (after the `max-width: 600px` block, before `</style>` at line 413), so it wins the cascade over both existing caps:

```css
    /* Phones & tablets (touch input): the image spans the full content width.
       Without this, the "desktop short screen" rule above collapses it on a
       landscape phone (100vh - 16rem = ~184px cap) and the mobile rule caps
       portrait photos at 45vh, so the image never reaches the screen width.
       "FACE N" labels and the keyboard hotkey hints are unreadable/unusable
       on touch screens; desktop (hover + fine pointer) is unchanged. */
    @media (hover: none) and (pointer: coarse) {
        .image-wrapper { width: 100%; }
        .image-wrapper img { max-height: none; margin: 0 auto; }
        .image-wrapper:not(.show-labels) .face-label { display: none; }
        .hotkey-hints { display: none; }
    }
```

Why this shape:

- **Touch detection by input capability.** `(hover: none) and (pointer: coarse)` is true on phones in portrait, in landscape, *and* in Safari "Request Desktop Website" mode; false on any mouse-driven desktop. A width-based query misses landscape; a pixel-width threshold cannot reliably separate phones from tablets or desktops.
- **`width: 100%` on the wrapper** makes the image area the full content width in every regime; **`max-height: none`** lifts both caps (mobile `45vh` and the short-screen `calc`), because this block is last in the stylesheet (equal specificity → document order wins). No `!important` needed (verified).
- **`margin: 0 auto`** centers images narrower than the content width (safety net; normally the image fills it).
- **`:not(.show-labels)`** implements the Revision 3 default: touch devices hide the chips unless the viewer explicitly turns them on. Every box keeps its `aria-label="Face N"`, its tap target and its behavior. Desktop keeps the labels unless toggled off (`.hide-labels` on the wrapper).
- **`.hotkey-hints`** lists keyboard shortcuts and `Face 1 … Face N` text; hidden on touch for consistency (it is already hidden in portrait, shown in landscape today). The keyboard *handlers* keep working on tablets with hardware keyboards; only the hint row disappears (decision D2).
- Bonus: on touch, the image no longer depends on `vh` at all, so iOS dynamic-toolbar `vh` variance can no longer move the image size.

### 3.1 Known behaviors and accepted limitations

1. **Sharpness in landscape on retina phones (decision D5):** at 844 CSS px × DPR 3 the image needs ~2532 device px from a 1280 px source ≈ 2× upscale (soft). Portrait phones need ~1212 device px, which is fine. Desktop is at ~1.32× today. Options in D5.
2. **Images narrower than the column** stay at natural size (no upscaling), centered by `margin: 0 auto`. Rare (thumbs are requested at 1280), and if it happens, side gutters exist only inside the wrapper; a draw started on a gutter yields out-of-range coordinates (rejected by the existing validation with "Bounding box out of allowed range"). Pre-existing class of behavior on small screens; can be hardened later if ever observed.
3. **Extreme panoramas** keep the base `min-height: 300px` on the wrapper, so a very short/wide image can leave some surface area below it (pre-existing on desktop; unchanged here; one-line tune possible later).
4. **Load-time layout shift:** pre-load the wrapper is a 300 px placeholder; a portrait photo on a landscape phone then grows to ~1266 px, moving the controls down while loading. Accepted for now; mitigable later by reserving space from the stored detection dimensions (`aspect-ratio`).

### 3.2 Label toggle (Revision 3)

- **Button:** "Hide labels" / "Show labels" next to "+ Add Missing Face", rendered only when the image has more than one face. `aria-pressed` mirrors the state and the label switches with it.
- **Shortcut:** `H` on multi-face pages (ignored while the manual-draw mode is active, and ignored on key repeat).
- **Persistence:** `localStorage["wikivisage-face-labels"]` = `"on"` / `"off"`, wrapped in `try/catch` so private mode still toggles for the page.
- **Default with no stored preference:** shown on hover/fine-pointer devices, hidden on `(hover: none) and (pointer: coarse)`, the Revision 2 behavior.
- **Cap:** only faces 1–9 get a visual `Face N` tag (the digit shortcuts stop at 9); every box keeps its `aria-label`, and the hotkey-hints row lists only the numbered faces that exist.
- **Single-face pages:** no tag, no button, no `H` handler; nothing to toggle.

## 4. Verified behavior after the fix (WebKit, Safari's engine)

Measured with the exact CSS block above; the replica runs without the site header (and the page's always-rendered flash-message gap), so the real page adds ≈140–150 px to the y-positions (horizontal numbers are unchanged):

| Regime (touch unless noted) | Before | After |
|---|---|---|
| iPhone landscape 956×440, landscape photo | img **276×184** (29 % of screen); controls y≈494, **already below the 440 px fold**; document 638 | img **844×562** (full content width); controls y≈765; document 829 |
| iPhone landscape 956×440, **portrait photo** | img **123×184** (13 %) | img **844×1266**; controls y≈1469 (≈2.3 screens down; decision D1); document 1533 |
| iPhone portrait 440×956, landscape photo | img 404×269 (already full width); labels visible | unchanged image; controls bottom ≈505, fits one screen; labels hidden |
| iPhone portrait 440×956, **portrait photo** | img **287×430** (65 %) | img **404×606** (full width); controls bottom ≈842 in the replica (≈990 on the real page, a short scroll); labels hidden |
| Phone in desktop-mode (980 wide) | — | img 844 (full width); labels hidden |
| **Desktop** 1024×2000 (mouse) | 844×562; labels visible | **identical**: 844×562; labels visible |
| **Desktop** 1024×800 short window (mouse) | 816×544; labels visible | **identical**: 816×544; labels visible |

Also verified: face boxes track the resized image in every case (a resize pass on the fixed layout moves box 0 from a 17×23 stub to 53×70 at the correct place), rotation re-flows correctly both ways, and no horizontal overflow appears in any probed case.

Revision 3 note: the `labels hidden` rows above are the **default** state on touch. With the toggle stored as `"on"` the chips render at the same positions; desktop behavior is unchanged unless the viewer hides them.

## 5. Staged plan

**Stage 1: Implement & verify locally (no deploy impact)**
1. Insert the CSS block in `templates/classify.html` (single hunk, before line 413).
2. Regenerate both replicas from the live templates and re-run the WebKit harness (§9); expected numbers: §4.
3. Desktop smoke check (Chrome and/or Safari): labels visible, window-resize behavior unchanged.
4. `ruff check .` + `pytest tests/ -q` (sanity; nothing here touches them).

**Stage 2: Ship & verify on device**
1. Ships with the next regular release (`deploy.yml`); template-only change; no migration, no worker/face-service implication.
2. On-device checklist (real iPhone):
   - Portrait: image spans the content width; **labels hidden by default** (the toggle shows them); tapping a face still classifies; no hotkey-hints row; with a portrait photo the buttons are at most a short scroll away.
   - Landscape: image spans the content width; scroll per decision D1; labels hidden by default, `H`/button toggles them and the choice survives a reload.
   - Rotate mid-page both ways: layout re-flows and **face boxes stay aligned with the faces**.
   - Android smoke test (real device or Chrome device emulation): full width, no labels, tap works.
   - Tablet check if decision D3 accepts tablets.
   - Desktop spot check: labels visible; short-window behavior unchanged.
3. Close the loop with the reporter. The acceptance criterion is the reporter's answer: if the tiny image **still reproduces upright** after the update, request the **original full-resolution screenshot** and re-open (§2.1).
   Optional pre-release preview: open `replica.html` from the phone over the LAN (`python3 -m http.server` in the harness folder) or use Safari Web Inspector against the deployed page.

**Stage 3: Optional follow-ups (only if wanted)**
- Edge-to-edge bleed (decision D4): recipe already verified (§6/D4).
- Sharper source via `srcset` 1920/3840 (decision D5).
- Pinned controls bar if landscape scrolling annoys (decision D1).
- `aspect-ratio` reservation to remove the load-time shift (§3.1-4).
- Copy pass: the hint texts still say *press "None"* / *"Press Esc to cancel"* on touch, and keyboarded tablets lose the hotkey hints (D2 nuance); needs translatable strings, so a separate change.

## 6. Decisions needed

1. **Landscape scrolling.** After the fix, a landscape phone scrolls to reach the buttons: landscape photo → controls at y≈905 on the real page (replica 765 + ≈140 px of site chrome) in a 440 px viewport ≈ 1.2 viewport-heights of scrolling; portrait photo → y≈1609 (replica 1469 + chrome) ≈ 2.8 viewport-heights. Note the pre-fix state was **also** scrolled (controls y≈494); the "everything fits without scroll" intent was already broken on landscape phones. Options: **(a) accept the scroll** (recommended: simplest, matches "full width"); (b) pin the controls as a sticky bottom bar on touch (extra styling + its own verification); (c) a binding height cap if the scroll ever feels extreme (e.g. `250vh` → the 2:3 photo renders 733×1100 = 87 % of the content width).
2. **Hotkey-hints row on touch: hidden** (already included in the block). Keep unless you object. (Note: on an iPad with a hardware keyboard the shortcuts still work; only the hints are hidden.)
3. **Scope: include tablets?** Pure CSS cannot separate an iPad from a phone. Recommended: **yes**. The same touch rationale applies. This is also the only input needed from you: do you classify on a tablet?
4. **Container gutter (88 % / 92 %) vs true edge-to-edge.** Both variants are **verified**: base fix = image 844 px in the 852 px content column (88 % of the 956 px screen; 404/440 = 92 % in portrait). Edge-to-edge recipe (wrapper spans the full viewport, no horizontal overflow, border thinned to 2 px): `width: 100vw; max-width: 100vw; margin-left: calc(50% - 50vw); margin-right: calc(50% - 50vw);` → image 952 px wide in landscape (100 %). Mocks to compare (open both): `r3-L-fix.png` vs `r5-ee-956-land.png`, and `r3-P1-fix.png` vs `r5-ee-440-land.png`. **Recommendation: ship the container-gutter version first** (consistent with the site's edge rhythm; edge-to-edge can be added later in one line if it still feels narrow). Caveat if edge-to-edge is chosen: the wrapper's border/side shadow style needs a small aesthetic pass, and `100vw` assumes overlay scrollbars (true on phones); if edge-to-edge is adopted, verify the bleed on the real device; the harness renders a plain viewport (no notch/safe-area handling; the template sets no `viewport-fit=cover`).
5. **Sharpness on retina phones in landscape** (§3.1-1): **(a) accept the 1280 px source** and judge on the device (recommended first step); (b) switch this page's thumbnail to the 1920 step only, a one-line change that roughly halves the upscale at a moderate download increase; (c) full `srcset` with 1920/3840 steps. Note that a DPR-3 landscape phone needs ≈2532 device px, so it resolves to the **3840** file (multi-MB downloads); (d) any of these as a follow-up after the on-device check.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cascade order: a future rule added after this block could re-cap the image | The block carries a comment naming what it overrides; the harness re-verifies after any CSS edit once the replicas are regenerated (§9). |
| `hover`/`pointer` unsupported (very old browsers) | The block is ignored, so behavior falls back to today's; no regression. Modern iOS/Android browsers support it. |
| Hybrid touch laptops in tablet posture match the query | Working as intended (primary input is touch); attaching a mouse restores the desktop view. |
| iPad with hardware keyboard loses the hint row | Accepted (D2); the keyboard handlers continue to work. |
| Hidden labels remove a small tap surface above each box | The label is a child of `.face-box`; the box's tap target and geometry are unchanged (verified). |
| Tablets get the touch treatment (D3) | Deliberate; narrowing the query later is a one-line change. |
| Soft image on DPR-3 phones in landscape (≈2× upscale of the 1280 thumb) | Decision D5; the srcset follow-up is isolated and additive. |
| Load-time layout shift grows for portrait photos on landscape phones | Accepted (§3.1-4); `aspect-ratio` follow-up available. |
| Verification pre-ship is emulation only | The emulation is WebKit (the phone's engine) and pixel-pins the bug; Stage 2 adds a real-device checklist and an optional LAN preview. |

## 8. Rollback

Delete the `@media` block (§3). Revision 3's toggle is additive: remove the button, the labels script and the `:not(.show-labels)` selector to return to the Revision 2 behavior. No state, no migration, no service interaction (the `localStorage` key is inert if the script is gone).

## 9. Evidence & artifacts

The measurements in §4 came from a temporary WebKit harness that extracted the
CSS from the **live** templates and rendered a replica page, so a regenerated
run always tested current CSS. The harness (`build_replica.py` plus the
`run_webkit*.py` scripts) and its screenshots were session-local evidence and
are **not committed**; the filenames below are references, not reproducible
artifacts. If the harness is needed again, add it to `scripts/` with a
dev-only Playwright dependency.

- `replica.html`, `replica-portrait.html`: faithful page replicas; `nordic-pms-1280.jpg`: the same Commons thumb as in the report.
- Screenshots: `r3-L-base.png` (reproduces the report) · `r3-L-fix.png` (fixed, container width) · `r5-ee-956-land.png`, `r5-ee-440-land.png` (edge-to-edge variant) · `r3-P1-fix.png`, `r3-P2-*.png` (portrait phone).
- `classify-iphone-0.jpg` + crops: the derivative screenshot extract used for §2.1.
