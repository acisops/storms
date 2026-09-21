---
name: verify-storms
description: Verify changes to the storms package before considering them done — runs ruff lint/format checks, then walks through the manual data-driven sanity check this repo relies on in place of automated tests. Use after editing storms/ code, before committing.
---

This repo has no automated test suite, so verification is a combination of static checks and a manual sanity run against real data.

1. Run static checks and report any failures:
   ```
   ruff format --check
   ruff check
   ```
   If either fails, offer to run `ruff format` / `ruff check --fix` to resolve it, then re-check.

2. Manually sanity-check the actual behavior of the change:
   - If the change touches `storms/solar_wind.py` or plotting logic, run the relevant `SolarWind` method (or an existing/new notebook cell in `dev/` or `scripts/`) against real data on the ACIS ops filesystem, inside the activated Ska stack (`acisska`).
   - If the change touches a CLI app (`storms/apps/calc_p3_fluence.py`, `storms/apps/make_storm_plots.py`), run it directly with a realistic YAML config (see `doc/source/command_line.rst` for a sample) and confirm the output (plots, computed fluence, etc.) looks correct.
   - If the change touches `storms/txings_proxy/`, be aware that pretrained model artifacts under `Models/`/`Indices/` are not regenerated automatically — confirm whether the change requires retraining or is compatible with the existing artifacts.

3. Report what was checked and what was verified manually — since there's no automated coverage, be explicit about what still hasn't been exercised.
