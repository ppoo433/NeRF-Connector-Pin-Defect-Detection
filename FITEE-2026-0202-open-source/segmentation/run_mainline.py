"""Run the frozen PointNet++ MSG configuration on user-supplied inputs."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'infer'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--split-csv', type=Path, required=True)
    parser.add_argument('--train-run-id', help='Required for inference; a checkpoint run produced by training.')
    parser.add_argument('--sample-ids', default='test', help='test, trainval, all, or comma-separated integer IDs.')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20272100, help='Paper repeats used 20272100 and 20272101.')
    parser.add_argument('--epochs', type=int, default=50, help='Final experiment used 50 epochs.')
    args = parser.parse_args()
    if not args.data_dir.is_dir() or not args.split_csv.is_file():
        parser.error('Provide an existing data directory and split CSV.')
    if args.stage == 'infer' and not args.train_run_id:
        parser.error('--train-run-id is required for inference.')
    values = {
        'DAIMA_POINT_CLOUD_DIR': str(args.data_dir.resolve()),
        'DAIMA_SPLIT_CSV': str(args.split_csv.resolve()),
        'DAIMA_PN2_MSG_USE_RGB': '1', 'DAIMA_PN2_MSG_USE_DENSITY': '1',
        'DAIMA_PN2_MSG_USE_COLOR_CONSTANCY': '0', 'DAIMA_PN2_MSG_USE_NORMALS': '0',
        'DAIMA_PN2_MSG_USE_GLOBAL_XYZ': '0', 'DAIMA_PN2_MSG_DENSITY_K': '4',
        'DAIMA_PN2_MSG_DENSITY_K2': '32', 'DAIMA_PN2_MSG_DENSITY_MODE': 'knn_mean_std',
        'DAIMA_PN2_MSG_DENSITY_NN_MEDIAN_MODE': 'random',
        'DAIMA_PN2_MSG_DENSITY_SEED_SAMPLING_MODE': 'uniform',
        'DAIMA_PN2_MSG_DENSITY_LOSS_REWEIGHT_MODE': 'none',
        'DAIMA_PN2_MSG_LOSS_TYPE': 'bce_pos_weight', 'DAIMA_PN2_MSG_POS_WEIGHT_MODE': 'fixed',
        'DAIMA_PN2_MSG_POS_WEIGHT_FIXED': '0.75', 'DAIMA_PN2_MSG_COLOR_DROPOUT_BLOCK_P': '0.05',
        'DAIMA_PN2_MSG_COLOR_DROPOUT_POINT_P': '0', 'DAIMA_PN2_MSG_COLOR_JITTER_SCALE': '0',
        'DAIMA_PN2_MSG_COLOR_JITTER_SHIFT': '0', 'DAIMA_PN2_MSG_COLOR_JITTER_NOISE_STD': '0',
        'DAIMA_PN2_MSG_TAIL_NEG_LAMBDA': '0', 'DAIMA_PN2_MSG_NP_LAMBDA': '0',
        'DAIMA_PN2_MSG_ENABLE_CLOUD_GATE_EVAL': '0', 'DAIMA_PN2_MSG_BLOCK_POINTS': '4096',
        'DAIMA_PN2_MSG_INFER_BLOCK_BATCH_SIZE': '1', 'DAIMA_PN2_MSG_INFER_ALLOW_DRIFT': '0',
        'DAIMA_PN2_MSG_RANDOM_SEED': str(args.seed), 'DAIMA_PN2_MSG_FOLD_ID': str(args.fold),
        'DAIMA_PN2_MSG_EPOCHS': str(args.epochs),
        'DAIMA_PN2_MSG_INFER_SAMPLE_IDS': args.sample_ids,
    }
    if args.train_run_id:
        values['DAIMA_PN2_MSG_TRAIN_RUN_ID'] = args.train_run_id
    os.environ.update(values)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / 'dl_pn2_msg' / f'{args.stage}_pn2_msg.py')]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
