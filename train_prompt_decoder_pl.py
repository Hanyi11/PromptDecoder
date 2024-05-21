import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from ObjectDetectionDatamodule_pl import ObjectDetectionDataModule
#from prompt_decoder_autoreg_pl import PromptDecoder
from prompt_decoder_autoreg_pl_without_teacher_forcing import PromptDecoder
import wandb

def train(args) -> None:
    """
    Sets up and trains the object detection model using the PromptDecoder and ObjectDetectionDataModule.

    Args:
    args (Namespace): Parsed arguments containing training configurations.
    """

    # Print all parameters before training
    print("Training Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Initializes the data module with training and testing data paths and batch size.
    data_module = ObjectDetectionDataModule(train_dir=args.train_dir, val_dir=args.val_dir, batch_size=args.batch_size)

    # Creates the model with specified configurations like learning rate and architecture parameters.
    model = PromptDecoder(
        learning_rate=args.learning_rate, 
        loss_type=args.loss_type, 
        max_epochs=args.max_epochs, 
        max_num_coord=args.max_num_coord, 
        transformer_dim=args.transformer_dim, 
        nheads=args.nheads, 
        dim_feedforward=args.dim_feedforward, 
        num_layers=args.num_layers, 
        pre_norm=args.pre_norm
    )

    # Initializes a CSV logger to record training progress into a CSV file at specified directory.
    csv_logger = pl_loggers.CSVLogger(args.log_dir)

    # Initializes the Wandb logger
    wandb_logger = pl_loggers.WandbLogger(
        name=f"{args.project_name}_{args.sub_name}",
        project=args.project_name,
        log_model=True,
        save_dir=args.log_dir
    )

    # Prepares a unique checkpointing name combining project and subproject names.
    checkpointing_name = f"{args.project_name}_{args.sub_name}"

    # Configures model checkpointing to save all models every 50 epochs with specific filename format.
    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=50,
        dirpath="checkpoints/",
        filename=f"{checkpointing_name}-{{epoch}}-{{val_loss:.2f}}",
        verbose=True
    )

    # Sets up a monitor to log learning rate changes at each epoch.
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    # Configures the PyTorch Lightning trainer with loggers and callbacks, including gradient clipping.
    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        callbacks=[checkpoint_callback_regular, lr_monitor],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val  # Add gradient clipping here
    )

    # Starts the model training process with the defined model and data module.
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an object detection model using PromptDecoder.')

    parser.add_argument('--train_dir', type=str, required=True, help='Directory containing the training dataset.')
    parser.add_argument('--val_dir', type=str, required=True, help='Directory containing the validation dataset.')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of samples in each batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for the optimizer.')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'smooth_l1'], help='Type of loss function to use.')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs for training.')
    parser.add_argument('--max_num_coord', type=int, default=2000, help='Maximum number of coordinates.')
    parser.add_argument('--transformer_dim', type=int, default=256, help='Dimension of transformer embeddings.')
    parser.add_argument('--nheads', type=int, default=4, help='Number of heads in the multihead attention mechanism.')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dimension of the feedforward network in transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the transformer.')
    parser.add_argument('--pre_norm', type=bool, default=True, help='Whether to use pre-normalization in layers.')
    parser.add_argument('--project_name', type=str, default='ObjectDetection', help='Name of the project for logging purposes.')
    parser.add_argument('--sub_name', type=str, default='default', help='Sub-name for detailed identification of the checkpoint.')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to store logs.')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clipping value.')

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.project_name, name=f"{args.project_name}_{args.sub_name}")

    train(args)

    # Finish the wandb run
    wandb.finish()
