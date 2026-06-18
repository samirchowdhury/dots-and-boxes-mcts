# Static Bot Tools

Export a trained EpsilonZero checkpoint into static browser assets:

```bash
uv run python static_bot/tools/export_checkpoint.py \
  runs/ez-flywheel/ez-policy-value-4x4-iter542-sims2000.npz \
  --set-latest
```

The static site loads `public/assets/bots/latest.json`, then fetches the pointed
`manifest.json` and `weights.bin`. Training artifacts in `runs/` are not served
directly.
