# MAMR scale recovery

Modified COLMAP Scaler used for marker-anchored metric reconstruction. The source includes RANSAC-gated triangulation, quality-weighted marker scale estimates, and IQR-filtered fusion. Upstream: https://github.com/lyehe/colmap_scaler. The upstream Apache-2.0 license is retained in `LICENSE`.

Use Python >=3.12 and the dependencies in `pyproject.toml`, in a separate environment. After installing this package, run:

```text
colmap-scaler --input YOUR_COLMAP_DATASET_COPY --marker-size YOUR_MARKER_SIDE_IN_METRES --export-json
```

Input layout: `images/` and `sparse/0/`. The scaler modifies the supplied reconstruction with a backup; use a working copy. Outputs use metres. COLMAP and Nerfacto are external upstream tools and are not vendored here. No images, reconstructions, or trained NeRF models are included.
