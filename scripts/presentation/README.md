# Presentation source

The editable deck is [relational-memory-proposal-v12.pptx](../../output/presentations/relational-memory-proposal-v12.pptx). Charts and tables are native PowerPoint objects. There are 21 slides: 14 main, 7 backup.

`build-proposal.mjs` uses the Codex presentations skill and `@oai/artifact-tool`. The v11 design reference is in `output/presentations/archive/`. This authoring runtime is separate from the simulation.

To rebuild in Codex, load the presentations skill and workspace dependencies, then follow its operation-marker and finalization workflow. Set `PRESENTATIONS_SKILL_DIR`, `RUNTIME_PYTHON`, `RUNTIME_NODE_MODULES` and `RUNTIME_BIN_DIR` to that machine's bundled runtime. Link this directory's ignored `node_modules` to the returned Node package directory. Run with the returned Node executable:

```bash
node scripts/presentation/build-proposal.mjs --draft-only
```

The draft and notes JSON go into ignored `.proposal-build/`. Omit `--draft-only` to finalize. The finalizer refuses to overwrite an existing final output; use a new version filename for future revisions. Render and review every slide before delivery.

The builder consumes `output/analysis/pinwheel-walkthrough.json` and historical `proposal-examples-v10.json`. The latter supplies the separate H2 learning curve; the bell example is no longer the main example. Reproduce with:

```bash
code/terralingua/.venv/bin/python scripts/pinwheel_walkthrough.py
code/terralingua/.venv/bin/python scripts/historical_bell_walkthrough.py
```

`export_presentation_text.py` extracts the final slide text, chart labels and notes using Python's standard library:

```bash
python3 scripts/presentation/export_presentation_text.py
```

PDF export uses LibreOffice. Animation remains in the PPTX/GIF; PDF pages are static. The standalone PNG/SVG in `output/analysis/` illustrates the same calculation; slide 8 uses an editable chart in the deck's original style.
