import argparse
from pprint import pprint
from typing import Literal, TypedDict

import torch
from lightning.pytorch import Trainer, seed_everything
from reaction_geognn.data_module import Wb97DataModule
from reaction_geognn.model import ProtoModel
from utils import abs_path, get_least_utilized_and_allocated_gpu

GRAPH_CACHE_PATH = abs_path('./cached_graphs/cached_wb97.bin')

def main():
    args = _parse_script_args()

    # To ensure deterministic
    seed_everything(0, workers=True)

    wb97_data_module = Wb97DataModule(
        fold_num = args['fold_num'],
        batch_size = args['batch_size'],
        cache_path = GRAPH_CACHE_PATH if args['cache_graphs'] \
            else None,
    )

    if args['precompute_only']:
        if args['cache_graphs']:
            print('"precompute-only" and "no-cache" shouldn\'t be used together. Else it\'ll not save the precomputed graphs, which is a waste of time.')
            return
        wb97_data_module.setup('fit')
        return

    # Use GPU.
    assert torch.cuda.is_available(), "No visible GPU."
    assert torch.cuda.device_count() > 1, "Only 1 GPU (expected multiple GPUs)."
    device = args['device'] if args['device'] \
        else get_least_utilized_and_allocated_gpu()

    model = ProtoModel(
        embed_dim = args['embed_dim'],
        gnn_layers = args['gnn_layers'],
        dropout_rate = args['dropout_rate'],
        out_size = 1,
        lr = args['lr'],
        _batch_size = args['batch_size'],
    )
    trainer = Trainer(
        deterministic = True,
        # disable validation when overfitting.
        limit_val_batches = 0 if args['overfit_one_batch'] else None,

        enable_checkpointing = args['load_save_checkpoints'],
        accelerator = device.type,
        devices = [device.index],
        overfit_batches = 1 if args['overfit_one_batch'] else 0,
        max_epochs = args['epochs'],
    )

    # disable validation doesn't seem to work with overfit_batches.
    # this should force it to work.
    if args['overfit_one_batch']:
        trainer.limit_val_batches = 0

    trainer.fit(model, datamodule=wb97_data_module)



class Arguments(TypedDict):
    # For debugging.
    precompute_only: bool
    load_save_checkpoints: bool
    cache_graphs: bool
    overfit_one_batch: bool

    # Model's hyper params.
    embed_dim: int
    dropout_rate: float
    gnn_layers: int
    lr: float
    device: torch.device | None

    # Trainer/Data module's params.
    batch_size: int
    epochs: int
    fold_num: Literal[0, 1, 2, 3, 4]

def _parse_script_args() -> Arguments:
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--precompute-only', default=False, action='store_true', help='precompute graph cache file only')
    parser.add_argument('--no-load-save', default=False, action='store_true', help='prevents loading/saving of checkpoints')
    parser.add_argument('--no-cache', default=False, action='store_true', help='prevents loading/saving/precomputing of graph cache file')
    parser.add_argument('--overfit-one-batch', default=False, action='store_true', help='train on 1 batch and disable validation to attempt to overfit')

    parser.add_argument('--embed-dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--gnn-layers', type=int, default=8, help='num of GNN layers')

    parser.add_argument('--fold-num', type=int, default=0, help='wb97xd3 fold_num-dataset to use')
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--device', type=str, default=None, help="device to run on")
    args = parser.parse_args()

    output: Arguments = {
        'precompute_only': args.precompute_only,
        'load_save_checkpoints': not args.no_load_save,
        'cache_graphs': not args.no_cache,
        'overfit_one_batch': args.overfit_one_batch,

        'embed_dim': args.embed_dim,
        'dropout_rate': args.dropout_rate,
        'gnn_layers': args.gnn_layers,
        'lr': args.lr,
        'device': torch.device(args.device) if args.device else None,

        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'fold_num': args.fold_num,
    }
    print('Arguments:')
    pprint(output)
    print('\n')
    if output['precompute_only']:
        print('Warning: Only precomputation of graph cache will be done.')
        return output
    if not output['load_save_checkpoints']:
        print('Warning: No loading/saving of checkpoints will be done.')
    if not output['cache_graphs']:
        print('Warning: No loading/saving/precomputing of graph cache file will be done.')
    print('\n')
    return output



if __name__ == "__main__":
    main()
