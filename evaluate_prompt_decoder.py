import torch
import gc
import pytorch_lightning as pl
from ObjectDetectionDatamodule_pl import ObjectDetectionDataModule
from prompt_decoder_autoreg_pl import PromptDecoder

def evaluate(
    checkpoint_path: str, 
    train_dir: str,
    val_dir: str, 
    batch_size: int = 4, 
    max_num_coord: int = 2000, 
    transformer_dim: int = 256, 
    nheads: int = 4, 
    dim_feedforward: int = 512, 
    num_layers: int = 3, 
    pre_norm: bool = True
) -> None:
    """
    Loads a checkpoint and evaluates it on the validation data.

    Args:
    checkpoint_path (str): Path to the checkpoint file.
    test_dir (str): Directory containing the test/validation dataset.
    batch_size (int, optional): Number of samples in each batch. Defaults to 32.
    max_num_coord (int, optional): Maximum number of coordinates. Defaults to 600.
    transformer_dim (int, optional): Dimension of transformer embeddings. Defaults to 256.
    nheads (int, optional): Number of heads in the multihead attention mechanism. Defaults to 8.
    dim_feedforward (int, optional): Dimension of the feedforward network in transformer. Defaults to 512.
    num_layers (int, optional): Number of layers in the transformer. Defaults to 3.
    pre_norm (bool, optional): Whether to use pre-normalization in layers. Defaults to True.

    Returns:
    None
    """

    # Initializes the data module with the testing data path and batch size.
    data_module = ObjectDetectionDataModule(train_dir, val_dir, batch_size)

    # Creates the model with specified configurations like architecture parameters.
    model = PromptDecoder(learning_rate=0, loss_type='mse', max_epochs=0, max_num_coord=max_num_coord, transformer_dim=transformer_dim, nheads=nheads, dim_feedforward=dim_feedforward, num_layers=num_layers, pre_norm=pre_norm)

    # Loads the checkpoint.
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    # Configures the PyTorch Lightning trainer.
    trainer = pl.Trainer()

    # Evaluates the model on the validation/test set.
    trainer.validate(model, data_module)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    train_dir = '/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train'
    val_dir = '/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/test'
    checkpoint_path = '/home/icb/hanyi.zhang/main_master_thesis/checkpoints_second_run/ObjectDetection_default-epoch=599-val_loss=0.00.ckpt'
    evaluate(checkpoint_path=checkpoint_path, train_dir=train_dir, val_dir=val_dir)