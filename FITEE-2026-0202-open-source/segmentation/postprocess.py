"""Frozen quantile (rho=0.425) + base OBB refinement, without legacy refiners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud', required=True, type=Path, help='PLY in metres, with the same point order as pin_prob.')
    parser.add_argument('--prediction', required=True, type=Path, help='NPZ with pin_prob from the inference stage.')
    parser.add_argument('--config', required=True, type=Path, help='OBB template configuration adapted to your own reference.')
    parser.add_argument('--output', required=True, type=Path, help='A NEW output directory.')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    params = config['params']
    if config.get('method') != 'quantile' or float(params.get('target_pin_frac', 0)) != 0.425:
        parser.error('The paper mainline requires quantile target_pin_frac=0.425.')
    if len(params.get('b4_base_boxes_m', [])) != 3:
        parser.error('Define exactly three OBBs in your reference frame; no private template geometry is distributed.')
    if not params.get('b4_base_box_enable') or not params.get('b4_reg_enable') or not params.get('b4_reg_require_success'):
        parser.error('The paper mainline requires OBB correction and successful template registration.')
    template_path = Path(params['b4_reg_template_path']).expanduser()
    if not template_path.is_absolute():
        template_path = args.config.resolve().parent / template_path
    if not template_path.is_file():
        parser.error('Provide an existing reference template PLY in b4_reg_template_path.')
    params['b4_reg_template_path'] = str(template_path.resolve())
    if args.output.exists():
        parser.error('--output must be a new directory.')
    import numpy as np
    from obb_core import _threshold_quantile, _b4_apply_manual_base_boxes
    from ply_utils import read_ply_vertex, write_ply_vertex_binary_little_endian

    vertex = read_ply_vertex(args.cloud)
    xyz = np.column_stack([vertex[c] for c in ['x','y','z']])
    with np.load(args.prediction, allow_pickle=False) as archive:
        probability = np.asarray(archive['pin_prob'], dtype=np.float32)
    if probability.ndim != 1 or len(probability) != len(xyz) or not len(probability) or not np.all(np.isfinite(probability)):
        raise ValueError('pin_prob must be finite, non-empty, and aligned one-to-one with the cloud.')
    threshold, debug = _threshold_quantile(probability, target_pin_frac=0.425)
    mask, obb_debug = _b4_apply_manual_base_boxes(pred_pin=probability >= threshold, points_xyz=xyz, params=params)
    args.output.mkdir(parents=True)
    for name, keep in [('pins',mask), ('base',~mask)]:
        write_ply_vertex_binary_little_endian(args.output / f'{name}.ply', vertex_props={k:v[keep] for k,v in vertex.items()})
    (args.output / 'report.json').write_text(json.dumps({'threshold':threshold, **debug, **obb_debug},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
