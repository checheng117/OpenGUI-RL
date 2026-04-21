# Contributing

OpenGUI-RL is a course/research artifact. Contributions are welcome when they preserve the scoped project framing.

## Good Contributions

- Bug fixes for dataset interfaces, metrics, or config loading.
- Clear documentation improvements.
- Lightweight tests for reward, metrics, candidate representation, and verifier logic.
- Reproduction notes that distinguish public assets from externally obtained benchmark data.

## Scope Guardrails

Please do not reframe the repository as:

- a full browser automation agent,
- an online PPO browser-training system,
- a benchmark redistribution package,
- evidence that reward optimization universally dominates supervision.

When adding results, include the exact benchmark split, data access assumptions, config, seed, and whether the artifact is redistributable.

## Local Checks

```bash
pytest
ruff check src scripts tests
```
