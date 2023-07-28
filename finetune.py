import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from reaction_geognn.data_modules import Wb97DataModule
from reaction_geognn.model import ReactionDownstreamModel
from utils import abs_path, get_least_utilized_and_allocated_gpu

GPU: torch.device | None = None
BATCH_SIZE = 50
LR = 3e-4
EPOCHS = 100

SEED = 0
GRAPH_CACHE_PATH = abs_path('cached_graphs/cached_wb97_superimposed.bin', __file__)
CHECKPOINT_PATH = abs_path('lightning_logs/version_119/checkpoints/epoch=66-std_val_loss=0.3939403295516968.ckpt', __file__)

def main():
    # To ensure deterministic
    seed_everything(SEED, workers=True)

    wb97_data_module = Wb97DataModule(
        fold_num = 0,
        batch_size = BATCH_SIZE,
        shuffle = True,
        cache_path = GRAPH_CACHE_PATH,
    )
    # Use GPU.
    assert torch.cuda.is_available(), "No visible GPU."
    assert torch.cuda.device_count() > 1, "Only 1 GPU (expected multiple GPUs)."
    device = GPU or get_least_utilized_and_allocated_gpu()

    model = ReactionDownstreamModel.load_from_checkpoint(
        CHECKPOINT_PATH,
        lr = LR,
        _logged_hparams = {
            'batch_size': BATCH_SIZE,
            'notes': 'finetune b97 -> wb97'
        },
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
        accelerator = device.type,
        devices = [device.index],
        max_epochs = EPOCHS,
    )

    trainer.fit(model, datamodule=wb97_data_module)



if __name__ == "__main__":
    main()
