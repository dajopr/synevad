# Dispatch

One subagent per category. Agents start cold, so everything the category needs
must be in the prompt — and nothing else, because whatever you put in the
prompt is what the firewall has to hold.

## Concurrency

Default 4 concurrent categories; `--concurrency N` overrides. Each agent runs a
literature search plus two skills, so it is slow and context-heavy — that is
exactly why it is a subagent and not an inline loop. Launch the next category
as soon as a slot frees; do not wait for the whole wave.

## Prompt template

Substitute the bracketed values from the category's manifest entry. Send it
verbatim otherwise.

`category` is always the dataset directory name — in the prompt, in the paths,
and in the taxonomy the agent writes. `category_description` rides alongside it
as extra grounding for material classification and literature search. Omit the
description line entirely when the category has none; an empty label invites an
agent to invent one.

```
Derive a standalone severity-graded defect taxonomy for one category, then
compile its generation prompts. Do both steps; do not stop after the first.

Category: <category>
<category_description, if present: "Category description: ...">
Benchmark: <benchmark>
Output folder: <output_root>/<category>/   (write your files there)

Use "<category>" verbatim as the `category` value in the taxonomy JSON — it is
the key every downstream consumer joins on. The description, if given, is
context for your analysis; you may echo it into an optional
`category_description` field, but it must never replace the category name.
Defect-free sample images — staged for you in <samples_dir>, and the only
images you may open:
  <sample_images, one per line>
<category_note, if present: "Note from visual inspection: ...">

Step 1 — invoke the `standalone-defect-taxonomy-derivation` skill and follow
it exactly. Write its JSON to:
  <outputs.taxonomy>

Step 2 — invoke the `standalone-defect-prompt-compiler` skill on that file and
write its JSON to:
  <outputs.prompts>

Contamination firewall — non-negotiable:
- Never search for, open, or reason about defective images or defect labels
  from this benchmark. Never combine the benchmark or category name with a
  defect term in a search query.
- Do not list, glob, or read any directory other than the samples folder above.
  Everything you need was copied there; the source dataset is off limits, and
  its sibling directories carry ground-truth defect labels in their names.
- Literature search is mechanism- and material-level only, per the derivation
  skill's rules. Log every query in `search_queries`.

Report back ONE line only, in this form — no taxonomy content, no prompt text,
no summary of the defect modes:
  <category>: <n> modes, <n> prompts, self_check=<pass|fail>, warnings=<n>
On failure report:
  <category>: FAILED — <one-sentence reason>
(Use "<category>" in that status line so it matches the manifest.)
```

## Handling results

- Parse the one-line status. Set `status` to `complete` on success, `failed`
  with the reason on failure, and write `run_manifest.json` back to disk after
  each category so the run is resumable at any point.
- Retry a failed category once, unchanged. A second failure is recorded, not
  retried again — repeated failure usually means a genuinely thin literature
  base for that material, which is a finding worth reporting.
- If an agent returns taxonomy content instead of a status line, ignore the
  content and read the artifact from disk instead. Never let a summary
  substitute for the file.
- Do not read the artifacts to check them — `scripts/validate_run.py` does that
  independently, which is the point: the checker must not be the thing that
  wrote them.

## Failure modes worth recognising

| Symptom | Cause | Action |
|---|---|---|
| `taxonomy.json` written but `prompts.json` missing | agent stopped after step 1 | re-dispatch step 2 alone (`--resume` marks it `taxonomy_done`) |
| Compiler reports "not a standalone taxonomy" | derivation emitted a chained schema | re-run derivation; do not hand-convert |
| Many categories fail with thin citations | corpus or search access is the bottleneck, not the category | stop the batch and tell the user before burning the remaining categories |
| Identical modes across unrelated categories | agent generalised instead of grounding | re-run those categories; note it in the report |
| Taxonomy's `category` is the description text | agent treated the description as the name | re-dispatch that category; `validate_run.py` flags this with a hint |
| Taxonomy invents a `category_description` that was never sent | description line was emitted empty instead of omitted | drop the line when absent; re-run that category |
| Agent reports it could not find the images | staging was skipped (`--no-stage`) or the run dir moved | re-scan without `--no-stage`; paths in the manifest must resolve from the agent's cwd |
