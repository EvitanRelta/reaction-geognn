import argparse, os
from pprint import pprint
from typing import Literal, TypedDict

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning_utils import LoggedHyperParams
from reaction_geognn.data_module import Wb97DataModule
from reaction_geognn.model import ProtoModel
from utils import LIGHTNING_LOGS_DIR, abs_path, \
    get_least_utilized_and_allocated_gpu

SEED = 0
GRAPH_CACHE_PATH = abs_path('cached_graphs/cached_wb97.bin', __file__)

def main():
    args = _parse_script_args()
    _validate_args(args)

    # To ensure deterministic
    seed_everything(SEED, workers=True)

    wb97_data_module = Wb97DataModule(
        fold_num = args['fold_num'],
        batch_size = args['batch_size'],
        shuffle = args['shuffle'],
        cache_path = GRAPH_CACHE_PATH if args['cache_graphs'] \
            else None,
    )

    if args['precompute_only']:
        wb97_data_module.setup('fit')
        return

    # Use GPU.
    assert torch.cuda.is_available(), "No visible GPU."
    assert torch.cuda.device_count() > 1, "Only 1 GPU (expected multiple GPUs)."
    device = args['device'] or get_least_utilized_and_allocated_gpu()

    # Hyper-params that's not used in the model, but is logged in the
    # lightning-log's `hparams.yaml` file.
    logged_hparams: LoggedHyperParams = {}
    if args['batch_size'] is not None:
        logged_hparams['batch_size'] = args['batch_size']
    if args['overfit_batches'] is not None:
        logged_hparams['dataset_size'] = args['batch_size'] * args['overfit_batches']
    if args['notes'] is not None:
        logged_hparams['notes'] = args['notes']

    model = ProtoModel(
        embed_dim = args['embed_dim'],
        gnn_layers = args['gnn_layers'],
        dropout_rate = args['dropout_rate'],
        out_size = 1,
        lr = args['lr'],
        _logged_hparams = logged_hparams,
    )
    trainer = Trainer(
        deterministic = True,
        # disable validation when overfitting.
        limit_val_batches = 0 if args['overfit_batches'] else None,

        enable_checkpointing = args['enable_checkpointing'],
        accelerator = device.type,
        devices = [device.index],
        overfit_batches = args['overfit_batches'],
        max_epochs = args['epochs'],
    )

    # disable validation doesn't seem to work with overfit_batches.
    # this should force it to work.
    if args['overfit_batches']:
        trainer.limit_val_batches = 0

    checkpoint_path = _get_checkpoint_path(args['resume_version'])
    trainer.fit(model, datamodule=wb97_data_module, ckpt_path=checkpoint_path)


class Arguments(TypedDict):
    # For debugging.
    precompute_only: bool
    enable_checkpointing: bool
    cache_graphs: bool
    overfit_batches: int
    notes: str | None

    # Model's hyper params.
    embed_dim: int
    dropout_rate: float
    gnn_layers: int
    lr: float

    # Trainer/Data module's params.
    fold_num: Literal[0, 1, 2, 3, 4]
    shuffle: bool
    batch_size: int
    epochs: int
    device: torch.device | None
    resume_version: int | None

def _parse_script_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Training Script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--precompute_only', default=False, action='store_true', help='precompute graph cache file only')
    parser.add_argument('--no_save', default=False, action='store_true', help='prevents saving of checkpoints')
    parser.add_argument('--no_cache', default=False, action='store_true', help='prevents loading/saving/precomputing of graph cache file')
    parser.add_argument('--overfit_batches', type=int, default=0, help='train on set number of batches and disable validation to attempt to overfit')
    parser.add_argument('--notes', type=str, default=None, help="notes to add to model's `hparams.yaml` file")

    parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gnn_layers', type=int, default=8, help='num of GNN layers')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--fold_num', type=int, default=0, help='wb97xd3 fold_num-dataset to use')
    parser.add_argument('--shuffle', default=False, action='store_true', help='enable shuffling on training dataset')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs to run')
    parser.add_argument('--device', type=str, default=None, help="device to run on")
    parser.add_argument('--resume_version', type=int, default=None, help="resume training from a lightning-log version")
    args = parser.parse_args()

    output: Arguments = {
        'precompute_only': args.precompute_only,
        'enable_checkpointing': not args.no_save,
        'cache_graphs': not args.no_cache,
        'overfit_batches': args.overfit_batches,
        'notes': args.notes,

        'embed_dim': args.embed_dim,
        'dropout_rate': args.dropout_rate,
        'gnn_layers': args.gnn_layers,
        'lr': args.lr,

        'fold_num': args.fold_num,
        'shuffle': args.shuffle,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'device': torch.device(args.device) if args.device else None,
        'resume_version': args.resume_version,
    }
    print('Arguments:')
    pprint(output)
    return output

def _validate_args(args: Arguments) -> None:
    print('\n')
    if args['precompute_only']:
        if not args['cache_graphs']:
            raise RuntimeError('"--precompute_only" and "--no_cache" shouldn\'t be used together. Else it\'ll not save the precomputed graphs, which is a waste of time.')
        if os.path.isfile(GRAPH_CACHE_PATH):
            raise RuntimeError(f'"--precompute_only" flag is used, but the cache file at "{GRAPH_CACHE_PATH}" already exists.')
        print('Warning: Only precomputation of graph cache will be done.')
        return

    if not args['enable_checkpointing']:
        print('Warning: No loading/saving of checkpoints will be done.')
    if not args['cache_graphs']:
        print('Warning: No loading/saving/precomputing of graph cache file will be done.')
    print('\n')

def _get_checkpoint_path(version_num: int | None) -> str | None:
    if version_num is None:
        return None

    checkpoint_dir = os.path.join(LIGHTNING_LOGS_DIR, f'version_{version_num}/checkpoints')
    checkpoint_file_names = os.listdir(checkpoint_dir)
    if len(checkpoint_file_names) == 1:
        return os.path.join(checkpoint_dir, checkpoint_file_names[0])

    checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')
    assert os.path.isfile(checkpoint_path), \
        f'Expected either 1 checkpoint file in "{checkpoint_dir}", ' \
        + f'or a last-checkpoint at "{checkpoint_path}", but neither is true.'
    return checkpoint_path



if __name__ == "__main__":
    main()
