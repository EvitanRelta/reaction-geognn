import argparse, os
from pprint import pprint
from typing import Literal, TypeAlias, TypedDict

import torch
from base_classes import LoggedHyperParams
from geognn import DownstreamModel, GeoGNNModel
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, \
    ModelCheckpoint
from reaction_geognn import B97DataModule, ReactionDownstreamModel, \
    Wb97DataModule
from utils import LIGHTNING_LOGS_DIR, abs_path, \
    get_least_utilized_and_allocated_gpu

DatasetNames: TypeAlias = Literal['wb97xd3', 'b97d3']

SEED = 0
GRAPH_CACHE_PATHS: dict[DatasetNames, str] = {
    'wb97xd3': abs_path('cached_graphs/cached_wb97_superimposed.bin', __file__),
    'b97d3': abs_path('cached_graphs/cached_b97_superimposed.bin', __file__),
}

def main():
    args = _parse_script_args()
    _validate_args(args)

    # To ensure deterministic
    seed_everything(SEED, workers=True)

    match args['dataset']:
        case 'wb97xd3':
            data_module = Wb97DataModule(
                fold_num = args['fold_num'],
                batch_size = args['batch_size'],
                shuffle = args['shuffle'],
                cache_path = GRAPH_CACHE_PATHS[args['dataset']] \
                    if args['cache_graphs'] else None,
            )
        case 'b97d3':
            data_module = B97DataModule(
                fold_num = args['fold_num'],
                batch_size = args['batch_size'],
                shuffle = args['shuffle'],
                cache_path = GRAPH_CACHE_PATHS[args['dataset']] \
                    if args['cache_graphs'] else None,
            )
        case _:
            raise RuntimeError(f'Expected value of "--dataset" flag to be "wb97xd3" or "b97d3", but got "{args["dataset"]}".')

    if args['precompute_only']:
        data_module.setup('fit')
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

    if args['pretrained_encoder_chkpt_path']:
        encoder_downstream = DownstreamModel.load_from_checkpoint(
            args['pretrained_encoder_chkpt_path'],
            map_location=device,
        )
        encoder = encoder_downstream.encoder

        assert_error_msg = "Encoder's `{encoder_hparam}={encoder_value}` doesn't match the arg `--{arg_hparam} {arg_value}`. \n" \
            + "This is just a check to prevent unintentionally setting different hyper-params."
        assert encoder.embed_dim == args['embed_dim'], \
            assert_error_msg.format(encoder_hparam="embed_dim", encoder_value=encoder.embed_dim, arg_hparam="embed_dim", arg_value=args["embed_dim"])
        assert encoder.num_of_layers == args['gnn_layers'], \
            assert_error_msg.format(encoder_hparam="num_of_layers", encoder_value=encoder.num_of_layers, arg_hparam="gnn_layers", arg_value=args["gnn_layers"])

        assert encoder.dropout_rate == args['dropout_rate'], \
            assert_error_msg.format(encoder_hparam="dropout_rate", encoder_value=encoder.dropout_rate, arg_hparam="dropout_rate", arg_value=args["dropout_rate"]) \
            + f" \nIf this was intentional (ie. u want to set `dropout_rate={args['dropout_rate']}` on the head " \
            + f"while to the encoder has `dropout_rate={encoder.dropout_rate}`), simply comment out this assert."

        # Since we're only using the encoder in the checkpoint (and not the
        # checkpoint's model's head), removing reference to `encoder_downstream`
        # will hopefully free memory used by the old head.
        del encoder_downstream
    else:
        encoder = GeoGNNModel(
            embed_dim = args['embed_dim'],
            dropout_rate = args['dropout_rate'],
            num_of_layers = args['gnn_layers'],
        )

    if args['pretrained_chkpt_path']:
        model = ReactionDownstreamModel.load_from_checkpoint(
            args['pretrained_chkpt_path'],
            map_location = device,
            _logged_hparams = logged_hparams, # Update logged hyper-params dict.

            # Allow head's hyper-params to be changed.
            # (This does NOT change/affect encoder's hyper-params)
            lr = args['lr'],
            dropout_rate = args['dropout_rate'],
        )
        assert model.hparams.dropout_rate == args['dropout_rate']
        assert model.hparams.lr == args['lr']
        assert model.hparams.out_size == 1

        assert_error_msg = "Encoder's `{encoder_hparam}={encoder_value}` doesn't match the arg `--{arg_hparam} {arg_value}`. \n" \
            + "This is just a check to prevent unintentionally setting different hyper-params."
        assert model.encoder.embed_dim == args['embed_dim'], \
            assert_error_msg.format(encoder_hparam="embed_dim", encoder_value=model.encoder.embed_dim, arg_hparam="embed_dim", arg_value=args["embed_dim"])
        assert model.encoder.num_of_layers == args['gnn_layers'], \
            assert_error_msg.format(encoder_hparam="num_of_layers", encoder_value=model.encoder.num_of_layers, arg_hparam="gnn_layers", arg_value=args["gnn_layers"])

        assert model.encoder.dropout_rate == args['dropout_rate'], \
            assert_error_msg.format(encoder_hparam="dropout_rate", encoder_value=model.encoder.dropout_rate, arg_hparam="dropout_rate", arg_value=args["dropout_rate"]) \
            + f" \nIf this was intentional (ie. u want to set `dropout_rate={args['dropout_rate']}` on the head " \
            + f"while to the encoder has `dropout_rate={model.encoder.dropout_rate}`), simply comment out this assert."

    else:
        model = ReactionDownstreamModel(
            encoder = encoder,
            dropout_rate = args['dropout_rate'],
            out_size = 1,
            lr = args['lr'],
            _logged_hparams = logged_hparams,
        )

    callbacks: list[Callback] = []

    if args['enable_checkpointing']:
        # Saves last and top-20 checkpoints based on the epoch's standardized
        # validation RMSE.
        callbacks.append(
            ModelCheckpoint(
                save_top_k = 1,
                save_last = True,
                monitor = "std_val_loss",
                mode = "min",
                filename = "{epoch:02d}-{std_val_loss:.2e}",
            )
        )
    if args['early_stop']:
        callbacks.append(EarlyStopping(monitor="std_val_loss"))

    trainer = Trainer(
        callbacks = callbacks,
        deterministic = True,
        limit_val_batches = 0 if args['no_validation'] else None,
        enable_checkpointing = args['enable_checkpointing'],
        accelerator = device.type,
        devices = [device.index],
        overfit_batches = args['overfit_batches'],
        max_epochs = args['epochs'],
    )

    # disable validation doesn't seem to work with overfit_batches.
    # this should force it to work.
    if args['overfit_batches'] and args['no_validation']:
        trainer.limit_val_batches = 0

    checkpoint_path = _get_checkpoint_path(args['resume_version'])
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)


class Arguments(TypedDict):
    # For debugging.
    precompute_only: bool
    enable_checkpointing: bool
    cache_graphs: bool
    overfit_batches: int
    no_validation: bool
    notes: str | None

    # Model's hyper params.
    embed_dim: int
    dropout_rate: float
    gnn_layers: int
    lr: float

    # Trainer/Data module's params.
    dataset: DatasetNames
    fold_num: Literal[0, 1, 2, 3, 4]
    shuffle: bool
    batch_size: int
    epochs: int
    early_stop: bool
    device: torch.device | None
    resume_version: int | None
    pretrained_chkpt_path: str | None
    pretrained_encoder_chkpt_path: str | None

def _parse_script_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Training Script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--precompute_only', default=False, action='store_true', help='precompute graph cache file only')
    parser.add_argument('--no_save', default=False, action='store_true', help='prevents saving of checkpoints')
    parser.add_argument('--no_cache', default=False, action='store_true', help='prevents loading/saving/precomputing of graph cache file')
    parser.add_argument('--overfit_batches', type=int, default=0, help='train on set number of batches in an attempt to overfit')
    parser.add_argument('--no_validation', default=False, action='store_true', help='disable validation')
    parser.add_argument('--notes', type=str, default=None, help="notes to add to model's `hparams.yaml` file")

    parser.add_argument('--embed_dim', type=int, default=256, help='embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gnn_layers', type=int, default=3, help='num of GNN layers')
    parser.add_argument('--lr', type=float, default=3e-4, help="learning rate")

    parser.add_argument('--dataset', type=str, default="wb97xd3", help='reaction dataset to use. Either "wb97xd3" or "b97d3"')
    parser.add_argument('--fold_num', type=int, default=0, help='which fold-split in the wb97xd3/b97d3 dataset to use')
    parser.add_argument('--no_shuffle', default=False, action='store_true', help='disable shuffling on training dataset')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs to run')
    parser.add_argument('--early_stop', default=False, action='store_true', help="stop training early if validation loss doesn't decrease 3 times in a row")
    parser.add_argument('--device', type=str, default=None, help='device to run on (eg. "cuda:1" for GPU-1, "cpu" for CPU). If not specified, auto-picks the least utilized GPU')
    parser.add_argument('--resume_version', type=int, default=None, help="resume training from a lightning-log version")
    parser.add_argument('--pretrained_chkpt_path', type=str, default=None, help="checkpoint path of the pretrained downstream-model to load")
    parser.add_argument('--pretrained_encoder_chkpt_path', type=str, default=None, help="checkpoint path of the pretrained encoder's downstream-model to load")
    args = parser.parse_args()

    output: Arguments = {
        'precompute_only': args.precompute_only,
        'enable_checkpointing': not args.no_save,
        'cache_graphs': not args.no_cache,
        'overfit_batches': args.overfit_batches,
        'no_validation': args.no_validation,
        'notes': args.notes,

        'embed_dim': args.embed_dim,
        'dropout_rate': args.dropout_rate,
        'gnn_layers': args.gnn_layers,
        'lr': args.lr,

        'dataset': args.dataset,
        'fold_num': args.fold_num,
        'shuffle': not args.no_shuffle,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'early_stop': args.early_stop,
        'device': torch.device(args.device) if args.device else None,
        'resume_version': args.resume_version,
        'pretrained_chkpt_path': abs_path(args.pretrained_chkpt_path, __file__) \
            if args.pretrained_chkpt_path != None else None,
        'pretrained_encoder_chkpt_path': abs_path(args.pretrained_encoder_chkpt_path, __file__) \
            if args.pretrained_encoder_chkpt_path != None else None,
    }
    print('Arguments:')
    pprint(output)
    return output

def _validate_args(args: Arguments) -> None:
    print('\n')
    if args['precompute_only']:
        if not args['cache_graphs']:
            raise RuntimeError('"--precompute_only" and "--no_cache" shouldn\'t be used together. Else it\'ll not save the precomputed graphs, which is a waste of time.')
        if os.path.isfile(GRAPH_CACHE_PATHS[args['dataset']]):
            raise RuntimeError(f'"--precompute_only" flag is used, but the cache file at "{GRAPH_CACHE_PATHS[args["dataset"]]}" already exists.')
        print('Warning: Only precomputation of graph cache will be done.')
        return

    if (args['resume_version'] != None) \
        + (args['pretrained_chkpt_path'] != None) \
        + (args['pretrained_encoder_chkpt_path'] != None) > 1:
        raise RuntimeError('"--resume_version", "--pretrained_chkpt_path" and/or "--pretrained_encoder_chkpt_path" cannot be used together. Else idk which checkpoint to load.')

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
