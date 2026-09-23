---
name: "standalone-defect-taxonomy-derivation"
description: "Derive a reference-free defect taxonomy whose severity levels are produced by INDEPENDENT single-shot edits of the pristine image at four grades — minimal, slight, moderate, severe — rather than by iterative chained editing. Use whenever the user asks for a standalone/non-chained/single-shot severity taxonomy, four-level severity grading (minimal/slight/moderate/severe), or defect descriptions where each severity is generated directly from the defect-free image for benchmarks like MVTec AD or VisA. Prefer the chained `defect-taxonomy-derivation` skill instead when the user wants a defect that grows over successive edits (none→early→moderate→advanced)."
---

# Standalone Severity-Graded Defect Taxonomy Derivation

Derive a per-category defect taxonomy for anomaly-detection benchmarks from literature and physical failure mechanisms alone. The output feeds one machine-parsed consumer: a set of **standalone image-editing prompts**, four per defect mode, each applied independently to the same defect-free source image.

**Standalone, not chained.** Every prompt describes the defect's *final appearance* at its severity grade and is applied to the pristine image. No prompt may reference a previous edit, a previous stage, or a change relative to anything ("deepen the existing…", "extend further…" are forbidden). This is the single structural difference from the chained variant, and it propagates into every step below: stage descriptors must be self-contained and mutually distinguishable in absolute terms, because the generator never sees the neighbouring grades.

Emit the exact JSON schema in the **Output schema** section at the end of this file.

**Why reference-free matters:** the benchmark's real defect images and labels are held out to test whether literature-derived synthesis works for assets lacking real defect references. If benchmark defect information leaks into the derivation, the entire experiment is invalid. Treat the contamination firewall below as the highest-priority constraint.

## Input contract

Per category you receive:

| Input | Allowed use |
|---|---|
| Category name + benchmark (e.g. MVTec `hazelnut`) | Material/structure classification only |
| 3–10 **non-defective** sample images | Material classification, typical-location grounding, fixed-block calibration |
| Materials/manufacturing-defect literature (provided corpus, or search under the rules below) | Failure modes, signatures, severity grading |

You do not receive — and must not seek — any defective image or defect-type label from the benchmark. Validation mapping against real labels is a separate downstream run, not yours.

## Contamination firewall (applies throughout)

1. **Prohibited queries:** never combine a benchmark or category identifier with defect terms. Forbidden: `"MVTec" + anything`, `"VisA" + anything`, `"<category> anomaly/defect/dataset"`. Allowed: mechanism/material-level queries like `"brittle fracture nut shell morphology"`, `"scratch morphology anodized aluminum"`.
2. **Prohibited sources:** benchmark papers, dataset pages, anomaly-detection publications showing benchmark defects, image searches depicting damaged benchmark objects.
3. **Accidental exposure:** if a source turns out to show or name benchmark defects, discard it entirely and log the incident in `contamination_log`.
4. **No label anticipation:** name modes after mechanisms, never after benchmark labels you may know from pretraining.
5. **Audit trail:** log every executed query string in `search_queries`, even when clean.

Search only mechanism-level, peer-reviewed materials-science and manufacturing literature, failure-analysis handbooks, and standards (ASTM, ISO). Avoid blogs, forums, and computer-vision publications.

## Procedure

### 1. Classify category, select failure modes

From the category name and non-defective images, classify material family (metal / polymer / textile / food-organic / semiconductor / printed substrate), structure (textured surface vs. discrete object), and rigidity. List candidate modes from these mechanism families: surface abrasion/scratch, crack/fracture, contamination/foreign material, deformation/dent, missing/misplaced part, print/coating defect, corrosion/discoloration.

**Select exactly 3–5 modes**, ranked by (a) physical dominance — documented as a primary failure mode for this material class — and (b) visual distinctness — no two selected modes share the same dominant geometry AND texture. Break ties by literature support. Record rejections with a one-line reason in `rejected_modes`. A small orthogonal set beats an exhaustive one: it keeps prompts prescriptive and severity grades separable.

### 2. Distill visual signatures (with cross-referencing)

Fill every signature field for each mode: mechanism, geometry, texture, color/tone, boundary behavior, typical location, anti-patterns. All values must be literature-supported.

Citation rules — these exist because fabricated or weakly grounded citations silently corrupt the whole taxonomy:

- **≥2 independent sources per mode.** Independent = different author groups, neither primarily citing the other for the claim.
- **Retrievable identifiers only** (DOI, arXiv ID, stable URL) plus a **verbatim quote** (≤50 words) you actually read. Never cite from memory.
- **Field-level agreement:** `geometry`, `texture`, `color_tone` each need ≥2 agreeing sources. `location` and `boundary` may rest on one source if flagged in `single_source_fields`.
- **Conflicts:** prefer the source describing the physical mechanism in text over one inferring appearance from a single photograph. Record conflict and resolution in `conflicts`.
- **Single-source fallback:** if a mode has only one independent source overall, drop it and promote the next candidate; if it's dominant with no substitute, keep it flagged `single_source_mode: true`.

### 3. Grade into minimal / slight / moderate / severe

Four grades, anchored asymmetrically:

- **severe** and **moderate** come **directly from field-damage literature**, cited as above. These are the documented, established appearances of the defect.
- **slight** is derived by **attenuating the moderate descriptor only**. Set `derivation: "softened_moderate"`.
- **minimal** is derived by **attenuating the slight descriptor only**. Set `derivation: "softened_slight"`.

Attenuate along these axes and no others: **extent** (length, area, count), **contrast** (tonal difference from surround), **depth/relief**, **boundary sharpness**, **continuity** (continuous vs. broken/intermittent). Reduce; never introduce a feature that is absent from the grade above. Sub-threshold damage is under-documented in the literature, and inventing features there destroys the monotonic severity ordering the whole benchmark rests on. Attenuation is a transformation, not a claim, so `minimal` and `slight` need no citation — record which axes you moved in `attenuation_axes`.

**Monotonicity:** every feature at grade *n* must reappear, intensified, at grade *n+1*. The four descriptors form a nested sequence, not four variants.

**Separability without numbers (qualitative grading).** Because each grade is generated independently, adjacent grades collapse into the same image if their descriptors differ only in a hedge word. Do **not** repair this by inventing quantities — no percentages, millimetres, pixel counts, or fractions of the object; the literature does not support them and they read as false precision. Instead:

- Move **≥2 attenuation axes** between every adjacent pair, and say so in `attenuation_axes`.
- Use a **distinct qualitative anchor per grade** on the primary axis, drawn from a monotone lexical ladder rather than repeated intensifiers. Prefer concrete perceptual anchors ("barely perceptible under close inspection", "clearly visible but confined to one spot", "prominent and spanning the region", "dominating the surface with secondary damage") over `slightly / somewhat / very / extremely` chains.
- No two descriptors in a mode may share their head noun phrase *and* their primary qualifier.

### 4. Write the standalone prompt set

One prompt per grade — four per mode, each a self-contained instruction for editing the **defect-free source image** (never the output of another prompt), ≤60 words:

- Phrase as an **absolute description of the finished appearance** at that grade: "a short, faint, broken hairline confined to the shell ridge". Forbidden: any comparative or incremental phrasing — `deepen`, `extend`, `further`, `existing`, `previous`, `more than`, `already`, `continue`.
- Carry the grade's qualitative anchor into the prompt text, so the editor gets the same separability signal the descriptor encodes.
- Encode each anti-pattern as an explicit negative clause, in **all four** prompts.
- Include the `region_of_interest` from the signature's location field.
- State that exactly one defect instance (or one cluster, if the mechanism is inherently multi-site) is to be added.

### 5. Fixed instruction block (once per benchmark)

From the non-defective images, extract lighting (direction, softness, speculars), background (color, texture, uniformity), viewpoint (angle, framing, pose), and image character (resolution class, sharpness, color cast). Encode as one ≤50-word constraint paragraph appended to every prompt; it pins everything except the defect edit.

### Scoring is out of scope

This skill emits no scorer criteria and no verification prompts — generated images are gated by a separate, general-purpose image-edit scorer. Anti-patterns therefore live only in the prompt negatives, which makes it doubly important that every anti-pattern reaches every prompt.

## Output and self-check

Emit one JSON document per category, exactly matching the schema below. Descriptors ≤40 words; anti-pattern items ≤10 words; prompts ≤60 words.

Before emitting, run the self-check and record it in `self_check`:

- 3–5 modes; no two share dominant geometry AND texture.
- Every signature field filled; every cited claim has identifier + verbatim quote.
- `geometry`, `texture`, `color_tone` each backed by ≥2 independent agreeing sources (or flagged).
- All four grades present, in order `minimal, slight, moderate, severe`.
- `moderate` and `severe` carry `derivation: "literature"` with citations; `slight` is `softened_moderate`; `minimal` is `softened_slight`.
- Grades monotonic: no feature appears at a grade that is absent from the grade above it.
- Adjacent grades differ on ≥2 `attenuation_axes` and carry distinct qualitative anchors; no two share head noun phrase and primary qualifier.
- No numeric quantities anywhere in descriptors or prompts.
- Every prompt is absolute and self-contained: contains none of `deepen, extend, further, existing, previous, more than, already, continue`, and does not refer to another grade ("than at moderate severity"). Incidental adjectival use of a grade word — "slight separation of the fracture edges" — is fine.
- Every anti-pattern appears as a negative clause in all four prompts of its mode.
- Every prompt names its `region_of_interest`.
- No benchmark defect label/image/paper used; `contamination_log` complete.
- Output validates against the schema.

Fix failures and re-run; emit only with `"passed": true` or with failures explicitly listed. The JSON is also the reproducibility artifact: a second run from the same corpus should regenerate materially the same taxonomy.

---

## Output schema

Emit exactly this JSON structure, one document per category. Field limits: descriptors ≤40 words; anti-pattern list items ≤10 words; `edit` strings ≤60 words; fixed instruction block ≤50 words; citation quotes ≤50 words (verbatim from source).

```json
{
  "benchmark": "mvtec_ad",
  "category": "hazelnut",
  "grading_mode": "standalone",
  "severity_scale": ["minimal", "slight", "moderate", "severe"],
  "classification": {
    "material": "food-organic",
    "structure": "discrete_object",
    "rigidity": "rigid"
  },
  "fixed_instruction_block": "keep the diffuse top-down lighting, plain gray background, centered object pose, and image sharpness unchanged",
  "modes": [
    {
      "name": "shell_crack",
      "mechanism": "brittle fracture under compressive load at shell curvature",
      "signature": {
        "geometry": "thin dark hairline following surface contour",
        "texture": "clean split, no material loss",
        "color_tone": "darkened line, no halo",
        "boundary": "hard edge",
        "location": "high-curvature ridge",
        "anti_patterns": ["wide gouge", "discoloration halo"]
      },
      "citations": [
        {
          "id": "doi:10.xxxx/xxxxx",
          "supports": ["mechanism", "geometry", "texture"],
          "quote": "verbatim quote from the source, max 50 words"
        },
        {
          "id": "doi:10.yyyy/yyyyy",
          "supports": ["geometry", "color_tone"],
          "quote": "verbatim quote from the source, max 50 words"
        }
      ],
      "conflicts": [
        {
          "field": "boundary",
          "sources": ["doi:...", "doi:..."],
          "resolution": "kept mechanism-level description over single-photograph inference"
        }
      ],
      "single_source_mode": false,
      "single_source_fields": [],
      "stages": {
        "minimal": {
          "descriptor": "barely perceptible broken hairline on one short section of the ridge, tone almost matching the shell",
          "derivation": "softened_slight",
          "attenuation_axes": ["extent", "contrast", "continuity"]
        },
        "slight": {
          "descriptor": "clearly visible but confined thin dark line on one ridge section, edges soft, no separation",
          "derivation": "softened_moderate",
          "attenuation_axes": ["extent", "contrast"]
        },
        "moderate": {
          "descriptor": "prominent continuous dark hairline spanning the shell ridge with hard edges",
          "derivation": "literature"
        },
        "severe": {
          "descriptor": "branching open fissure dominating the ridge with slight edge separation",
          "derivation": "literature"
        }
      },
      "prompt_set": [
        {
          "stage": "minimal",
          "edit": "add one barely perceptible broken hairline on a short section of the shell ridge, almost matching the surrounding shell tone",
          "negatives": ["no material loss", "no discoloration halo"],
          "region_of_interest": "high-curvature ridge",
          "instance_count": "single"
        },
        {
          "stage": "slight",
          "edit": "add one clearly visible but confined thin dark line on a section of the shell ridge, with soft edges and no separation",
          "negatives": ["no material loss", "no discoloration halo"],
          "region_of_interest": "high-curvature ridge",
          "instance_count": "single"
        },
        {
          "stage": "moderate",
          "edit": "add one prominent continuous dark hairline with hard edges spanning the shell ridge",
          "negatives": ["no material loss", "no discoloration halo"],
          "region_of_interest": "high-curvature ridge",
          "instance_count": "single"
        },
        {
          "stage": "severe",
          "edit": "add one branching open fissure dominating the shell ridge, with slight separation of the fracture edges",
          "negatives": ["no discoloration halo", "no crumbling debris"],
          "region_of_interest": "high-curvature ridge",
          "instance_count": "single"
        }
      ]
    }
  ],
  "rejected_modes": [
    {"name": "corrosion", "reason": "not applicable to organic shell material"}
  ],
  "contamination_log": [],
  "search_queries": [
    "brittle fracture nut shell morphology"
  ],
  "self_check": {
    "passed": true,
    "failures": []
  }
}
```

### Field notes

- `grading_mode` is always `"standalone"`. It is the flag a compiler uses to distinguish this taxonomy from a chained one; a chained taxonomy has `prompt_chain` with `transition` keys instead of `prompt_set` with `stage` keys.
- `severity_scale` is fixed and ordered: `["minimal", "slight", "moderate", "severe"]`. `stages` and `prompt_set` must both cover exactly these four, in this order.
- `derivation` is `"literature"` (cited; `moderate` and `severe` only), `"softened_moderate"` (`slight` only), or `"softened_slight"` (`minimal` only).
- `attenuation_axes` is required on `minimal` and `slight`, absent on the cited grades. Allowed values: `extent`, `contrast`, `depth`, `boundary_sharpness`, `continuity`. At least two per attenuated grade.
- `citations[].supports` lists the signature fields the quote backs. Union across citations must cover `mechanism`, `geometry`, `texture`, `color_tone`; `geometry`/`texture`/`color_tone` each need ≥2 sources.
- `prompt_set[].edit` is an **absolute** description of the finished appearance, applied to the defect-free source image. It must not reference another grade or a prior edit.
- `prompt_set[].instance_count` is `"single"` or `"cluster"` — `"cluster"` only when the mechanism is inherently multi-site (e.g. pitting corrosion).
- `negatives` repeats the mode's `anti_patterns` (plus any grade-specific ones) in **every** entry of `prompt_set`; there is no scorer downstream to catch them.
- No numeric quantities are permitted in `descriptor` or `edit` strings — grading is qualitative by design.
- `contamination_log` entries: `{"incident": "...", "source": "...", "action": "discarded"}`. Empty list means no exposure occurred.
- `search_queries`: every executed query string, for firewall auditability.
- `self_check.failures`: list failed checklist items verbatim if `passed` is false.
- No `scorer_criteria` field exists in this schema. Verification is handled by a separate general-purpose image-edit scorer outside this pipeline.

The compiled prompts are produced from this JSON by the companion `standalone-defect-prompt-compiler` skill.
