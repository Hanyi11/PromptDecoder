import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.nn import MSELoss, SmoothL1Loss
from transformer_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from sklearn.metrics import precision_score, recall_score, f1_score

class PromptDecoder(pl.LightningModule):
    """
    A decoder module for transformer models, integrating cross-attention,
    self-attention, and feedforward network layers to decode spatial and channel-wise
    information from image embeddings into coordinate space.

    Attributes:
        transformer_dim (int): Dimensionality of the input and output tokens.
        nheads (int): Number of attention heads in each attention layer.
        dim_feedforward (int): Dimensionality of the feedforward network's hidden layer.
        num_layers (int): Number of layers in the decoder.
        pre_norm (bool): If True, applies normalization before other operations within the layers.
        coord_to_tokens (nn.Linear): Converts coordinates to tokens.
        transformer_self_attention_layers (nn.ModuleList): List of self-attention layers.
        transformer_cross_attention_layers (nn.ModuleList): List of cross-attention layers.
        transformer_ffn_layers (nn.ModuleList): List of feedforward network layers.
        tokens_to_coord (nn.Linear): Converts tokens back to coordinates.

    Parameters:
        transformer_dim (int): See Attributes.
        nheads (int): See Attributes.
        dim_feedforward (int): See Attributes.
        num_layers (int): See Attributes.
        pre_norm (bool): See Attributes.
    """
    def __init__(
            self,
            learning_rate: float,
            loss_type: str,
            max_epochs: int,
            max_num_coord: int,
            transformer_dim: int,
            nheads: int,
            dim_feedforward: int,
            num_layers: int,
            pre_norm: bool,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.loss_type = loss_type
        # Select the loss function based on the input parameter
        if self.loss_type == 'mse':
            self.loss_func = MSELoss()
        elif self.loss_type == 'smooth_l1':
            self.loss_func = SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        self.max_epochs = max_epochs
        self.max_num_coord = max_num_coord
        self.coord_to_tokens = nn.Linear(4, transformer_dim) # coordinates embedding, from coordinates of size 4 to transformer_dim
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.nheads = nheads
        self.tokens_to_coord = nn.Linear(transformer_dim, 4)
        self.training_step_outputs = []
        self.val_step_outputs = []

        for _ in range(num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=transformer_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=transformer_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=transformer_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
    def forward(
            self,
            target_coord, # Batch x num_coord x 4 -> will be fed into nn.Linear(4, transformer_dim) -> Batch x num_coord x 256
            image_embedding: Tensor,  # Batch x 256 x 64 x 64
            visual_reference_embedding: Tensor,  # Batch x 256 x 64 x 64
            source_mask: Tensor,
            target_mask: Tensor,
    ) -> Tensor:
        """
        Processes input tensors through the PromptDecoder architecture and returns decoded output coordinates.

        Args:
            target_coord (Tensor): Batch x num_coord x 4, target coordinates of variable length.
            image_embedding (Tensor): Batch x 256 x 64 x 64, embeddings from SAM image encoder.
            image_pos_embedding (Tensor): Batch x 256 x 64 x 64, positional encodings for image embeddings.
            visual_reference_embedding (Tensor): Batch x (num_crops x H x W) x 256, additional visual reference embeddings.
            source_mask (Tensor): Mask for the source embeddings.
            target_mask (Tensor): Mask for the target coordinates.

        Returns:
            Tensor: Decoded coordinates, transformed back to their original space
        """
        num_coord = target_coord.shape[1]
        target_coord_tokens = self.coord_to_tokens(target_coord) # Batch x num_coord x transformer_dim
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # Batch x (64 x 64) x 256
        visual_reference_embedding = visual_reference_embedding.flatten(2).permute(0, 2, 1)  # Batch x (64 x 64) x 256

        # Concatenate coordinates tokens and visual reference embedding, the tokens should contain posiitonal information like point_embedding in SAM
        tokens = torch.cat((visual_reference_embedding, target_coord_tokens), dim=1)  # Batch x N_queries x transformer_dim, where N_queries = (num_crops x H x W) + num_coord

        # Process through each layer of the transformer
        # Note: NO source mask! NO pos embedding for target and source! Only target mask!
        for i in range(self.num_layers):
            tokens = self.transformer_cross_attention_layers[i](target=tokens, source=image_embedding, target_pos=None, source_mask=source_mask, source_pos=None)
            tokens = self.transformer_self_attention_layers[i](target=tokens, target_mask=target_mask, target_pos=None)
            tokens = self.transformer_ffn_layers[i](target=tokens)

        # Decode only the last token back to coordinates
        last_tokens = tokens[:, -num_coord:, :]  # Take the last tokens from the sequence
        final_output_coordinate = self.tokens_to_coord(last_tokens)  # transform it to a 4-dimensional space

        return final_output_coordinate # the last box coordinates will be added to the input sequence in the next step, so the input sequence length will be increased by one

    def configure_optimizers(self):
        # The scheduler may be too simple, try to use warm-up
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / self.max_epochs))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Indicates that the scheduler step should be called on an epoch basis
                'frequency': 1        # Indicates that the scheduler should step every epoch
            }
        }
    
    def training_step(self, batch, batch_idx):
        image_embedding, visual_reference_embedding, target_coord = batch
        batch_size = image_embedding.shape[0]

        source_mask = None

        diagonal = visual_reference_embedding.shape[1] + 1
        num_queries = visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3] + target_coord.shape[1]
        target_mask = self.subsequent_mask(num_queries, diagonal)
        target_mask = target_mask.expand(batch_size * self.nheads, -1, -1).to(self.device)

        output_coord = self.forward(target_coord=target_coord, image_embedding=image_embedding, visual_reference_embedding=visual_reference_embedding, source_mask=source_mask, target_mask=target_mask)
        
        pad = torch.tensor([-1, -1, -1, -1], dtype=torch.float32).to(self.device)
        valid_mask = (target_coord != pad).all(dim=-1)
        valid_output_coord = output_coord[valid_mask]
        valid_target_coord = target_coord[valid_mask]
        
        # use full output without mask
        #loss = self.loss_func(valid_output_coord, valid_target_coord)
        loss = self.loss_func(output_coord, target_coord)
        iou = self.calculate_iou(valid_output_coord, valid_target_coord)
        
        self.training_step_outputs.append({'loss': loss, 'iou': iou})
        
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        
        # Concatenate IoU values
        all_ious = torch.cat([x['iou'] for x in self.training_step_outputs])
        avg_train_iou = all_ious.mean()
        
        # Recalculate precision, recall, and F1 from concatenated IoU values
        precision, recall, f1 = self.calculate_precision_recall_f1(all_ious)
        
        self.log('train_loss', avg_train_loss)
        self.log('train_iou', avg_train_iou)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_f1', f1)

        print('train_loss: ', avg_train_loss)
        print('train_iou: ', avg_train_iou)
        print('train_precision: ', precision)
        print('train_recall: ', recall)
        print('train_f1: ', f1)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image_embedding, visual_reference_embedding, target_coord = batch
        batch_size = image_embedding.shape[0]
        
        source_mask = None

        diagonal = visual_reference_embedding.shape[1] + 1
        num_queries = visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3] + target_coord.shape[1]
        target_mask = self.subsequent_mask(num_queries, diagonal)
        target_mask = target_mask.expand(batch_size * self.nheads, -1, -1).to(self.device)

        output_coord = self.forward(target_coord=target_coord, image_embedding=image_embedding, visual_reference_embedding=visual_reference_embedding, source_mask=source_mask, target_mask=target_mask)
        
        pad = torch.tensor([-1, -1, -1, -1], dtype=torch.float32).to(self.device)
        valid_mask = (target_coord != pad).all(dim=-1)
        valid_output_coord = output_coord[valid_mask]
        valid_target_coord = target_coord[valid_mask]
        
        # use full output without mask
        #val_loss = self.loss_func(valid_output_coord, valid_target_coord)
        val_loss = self.loss_func(output_coord, target_coord)
        iou = self.calculate_iou(valid_output_coord, valid_target_coord)
        
        self.val_step_outputs.append({'loss': val_loss, 'iou': iou})
        
        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()

        # Concatenate IoU values
        all_ious = torch.cat([x['iou'] for x in self.val_step_outputs])
        avg_val_iou = all_ious.mean()

        # Recalculate precision, recall, and F1 from concatenated IoU values
        precision, recall, f1 = self.calculate_precision_recall_f1(all_ious)

        self.log('val_loss', avg_val_loss)
        self.log('val_iou', avg_val_iou)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)

        print('val_loss: ', avg_val_loss)
        print('val_iou: ', avg_val_iou)
        print('val_precision: ', precision)
        print('val_recall: ', recall)
        print('val_f1: ', f1)

        self.val_step_outputs.clear()

    @staticmethod
    def calculate_iou(pred_boxes, true_boxes):
        """
        Calculate the Intersection over Union (IoU) for each pair of predicted and true boxes.
        Args:
            pred_boxes (Tensor): Predicted bounding boxes (num_valid_coords, 4).
            true_boxes (Tensor): True bounding boxes (num_valid_coords, 4).
        Returns:
            Tensor: IoU for each pair of boxes (num_valid_coords,).
        """
        # Calculate intersection
        xA = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        yA = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        xB = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        yB = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        # Calculate areas
        boxAArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        boxBArea = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

        # Calculate union
        unionArea = boxAArea + boxBArea - interArea

        # Calculate IoU
        iou = interArea / unionArea

        return iou

    @staticmethod
    def calculate_precision_recall_f1(iou, iou_threshold=0.5):
        pred_labels = (iou >= iou_threshold).cpu().numpy()
        true_labels = np.ones_like(pred_labels, dtype=bool)

        if len(pred_labels) == 0:
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(true_labels, pred_labels, zero_division=1)
            recall = recall_score(true_labels, pred_labels, zero_division=1)
            f1 = f1_score(true_labels, pred_labels, zero_division=1)

        return precision, recall, f1

    @staticmethod
    def subsequent_mask(size, diagonal):
        mask = torch.triu(torch.ones(1, size, size), diagonal=diagonal).bool()
        return mask