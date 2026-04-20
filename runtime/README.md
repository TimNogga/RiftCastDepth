# Runtime Workspace

This folder is the local workspace for runtime depth-fusion sweeps.

## Structure

- `configs/`: runtime config variants used for sweeps (`none`, `real`, `synthetic`).
- `outputs/`: exporter outputs per run (ignored by git).
- `logs/`: run logs per configuration (ignored by git).
- `export_all_bins_to_meshes.py`: converts all `vertices.bin` + `faces.bin` pairs under `outputs/` to `voxel_based.obj`.

## Typical workflow

1. Run exporter with one config from `configs/`.
2. Inspect logs in `logs/`.
3. Convert mesh bins to OBJ:

```bash
python runtime/export_all_bins_to_meshes.py
```

By default, `outputs/` and `logs/` are local-only and excluded from version control.