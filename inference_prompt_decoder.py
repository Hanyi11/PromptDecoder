import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from copy import deepcopy
from typing import Tuple
import matplotlib.patches as patches
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from ObjectDetectionDatamodule_pl import ObjectDetectionDataModule
from prompt_decoder_autoreg_pl import PromptDecoder
import torch


# Define the transformation functions
def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def inverse_coords(coords: np.ndarray, original_size: Tuple[int, int], target_length: int) -> np.ndarray:
    """
    Inverse transformation of coordinates from resized back to original.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (old_w / new_w)
    coords[..., 1] = coords[..., 1] * (old_h / new_h)
    return coords

def inverse_boxes(boxes: np.ndarray, original_size: Tuple[int, int], target_length: int) -> np.ndarray:
    """
    Inverse transformation of boxes from resized back to original.
    """
    boxes = inverse_coords(boxes.reshape(-1, 2, 2), original_size, target_length)
    return boxes.reshape(-1, 4)

def filter_labels(labels: np.ndarray) -> np.ndarray:
    """
    Filter out the first box which is all zeros and all boxes that are (-1, -1, -1, -1).
    """
    # Remove the first box if it is all zeros
    if np.all(labels[0] == 0):
        labels = labels[1:]

    # Remove boxes that are (-1, -1, -1, -1)
    labels = labels[~np.all(labels == -1, axis=1)]
    
    return labels

def decode_coord(model, img_emb, ref_emb, max_len, start_symbol):
    """
    Autoregressive decoding of coordinates using the given model.

    Args:
        model (PromptDecoder): The trained PromptDecoder model.
        img_emb (numpy.ndarray): Image embeddings.
        ref_emb (numpy.ndarray): Visual reference embeddings.
        max_len (int): Maximum length of the coordinate sequence to be decoded.
        start_symbol (Tensor): The start symbol for the coordinate sequence.

    Returns:
        numpy.ndarray: Decoded coordinates.
    """
    device = next(model.parameters()).device
    model.eval()

    # Convert numpy arrays to PyTorch tensors and move them to the appropriate device
    img_emb = torch.tensor(img_emb, dtype=torch.float32).to(device)
    img_emb = img_emb.unsqueeze(0)
    ref_emb = torch.tensor(ref_emb, dtype=torch.float32).to(device)
    ref_emb = ref_emb.unsqueeze(0)

    # Initialize the sequence with the start symbol
    decoded_coords = [start_symbol]

    with torch.no_grad():
        for _ in range(max_len):
            # Prepare the input for the model
            target_coord = torch.stack(decoded_coords).unsqueeze(0).to(device)  # Shape: (1, len(decoded_coords), 4)

            # No source mask for simplicity
            source_mask = None

            # Prepare target mask based on the current sequence length
            batch_size = 1  # During inference, batch size is 1
            diagonal = ref_emb.shape[1] + 1
            num_queries = ref_emb.shape[2] * ref_emb.shape[3] + target_coord.shape[1]
            target_mask = model.subsequent_mask(num_queries, diagonal)
            target_mask = target_mask.expand(batch_size * model.nheads, -1, -1).to(device)

            # Decode the next coordinate
            output_coord = model(
                target_coord=target_coord,
                image_embedding=img_emb,
                visual_reference_embedding=ref_emb,
                source_mask=source_mask,
                target_mask=target_mask
            )

            # Get the last predicted coordinate
            next_coord = output_coord[0, -1, :]  # Shape: (4,)

            # Append the predicted coordinate to the sequence
            decoded_coords.append(next_coord)

    # Stack the decoded coordinates and return as a numpy array
    decoded_coords = torch.stack(decoded_coords).cpu().numpy()  # Convert to numpy array
    return decoded_coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an object detection model using PromptDecoder.')

    parser.add_argument('--train_dir', type=str, default='/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train_full', help='Directory containing the training dataset.')
    parser.add_argument('--val_dir', type=str, default='/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/test', help='Directory containing the validation dataset.')
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

    # File paths
    index = '00001'
    example_image_path = '/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train_full/images/cell_'+index+'.png'
    labels_path = '/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train_full/labels/cell_'+index+'_label.npy'

    # Read the image
    image = Image.open(example_image_path)

    # Get the original size of the image
    original_size = image.size  # (width, height)
    original_size = (original_size[1], original_size[0])  # Convert to (height, width)

    # Read the labels
    labels = np.load(labels_path)

    # Filter the labels
    filtered_labels = filter_labels(labels)

    # Inverse the transformation
    target_length = 1024
    original_boxes = inverse_boxes(filtered_labels, original_size, target_length)
    print('Number of objects: ', len(original_boxes))
    print('Labels:')
    print(original_boxes)

    # Plot the image with bounding boxes
    plt.imshow(image)
    ax = plt.gca()

    for box in original_boxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

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

    # Loads the checkpoint.
    checkpoint_path = '/home/icb/hanyi.zhang/main_master_thesis/checkpoints_second_run/ObjectDetection_default-epoch=599-val_loss=0.00.ckpt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    #print(model)
    
    # RUN THIS FOR DECODING COORDINATES
    img_emb = np.load('/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train/img_emb/cell_'+index+'.npy')
    ref_emb = np.load('/home/icb/hanyi.zhang/main_master_thesis/NeurIPS22-CellSeg-png/train/ref_crop_emb/crop_cell_'+index+'.npy')
    start_symbol = torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # Define an appropriate start symbol
    decoded_coords = decode_coord(model, img_emb, ref_emb, max_len=200, start_symbol=start_symbol)
    print(decoded_coords)