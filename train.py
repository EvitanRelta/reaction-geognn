import argparse, os
from pprint import pprint
from typing import TypedDict

import torch
from base_classes import LoggedHyperParams
from geognn import DownstreamModel, GeoGNNModel
from geognn.data_modules import QM9DataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import LIGHTNING_LOGS_DIR, abs_path, \
    get_least_utilized_and_allocated_gpu

SEED = 0
GRAPH_CACHE_PATH = abs_path('cached_graphs/cached_qm9.bin', __file__)

def main():
    args = _parse_script_args()
    _validate_args(args)

    # To ensure deterministic
    seed_everything(SEED, workers=True)

    qm9_data_module = QM9DataModule(
        task_column_name = ['gap', 'h298'],
        batch_size = args['batch_size'],
        shuffle = args['shuffle'],
        cache_path = GRAPH_CACHE_PATH \
            if args['cache_graphs'] else None,
        random_split_seed = SEED,
    )

    if args['precompute_only']:
        qm9_data_module.setup('fit')
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

    encoder = GeoGNNModel(
        embed_dim = args['embed_dim'],
        dropout_rate = args['dropout_rate'],
        num_of_layers = args['gnn_layers'],
    )
    model = DownstreamModel(
        encoder = encoder,
        task_type = 'regression',
        out_size = 2,  # Predicting QM9's HOMO-LUMO gap and enthalpy at 298K.
        num_of_mlp_layers = 3,
        mlp_hidden_size = 4 * args['embed_dim'],
        dropout_rate = args['dropout_rate'],
        lr = args['lr'],
        _logged_hparams = logged_hparams,
    )

    # Saves last and top-20 checkpoints based on the epoch's standardized
    # validation RMSE.
    chkpt_callback = ModelCheckpoint(
        save_top_k = 20,
        every_n_epochs = 5,
        save_last = True,
        monitor = "std_val_loss",
        mode = "min",
        filename = "{epoch:02d}-{std_val_loss:.2e}",
    )
    trainer = Trainer(
        callbacks = [chkpt_callback],
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
    trainer.fit(model, datamodule=qm9_data_module, ckpt_path=checkpoint_path)


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

    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gnn_layers', type=int, default=3, help='num of GNN layers')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")

    parser.add_argument('--no_shuffle', default=False, action='store_true', help='disable shuffling on training dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs to run')
    parser.add_argument('--device', type=str, default=None, help='device to run on (eg. "cuda:1" for GPU-1, "cpu" for CPU). If not specified, auto-picks the least utilized GPU')
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

        'shuffle': not args.no_shuffle,
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
