import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn, Tensor
from torch.nn import MSELoss, SmoothL1Loss
from transformer_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer
from sklearn.metrics import precision_score, recall_score, f1_score
import gc
from torch.utils.checkpoint import checkpoint

class PromptDecoder(pl.LightningModule):
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
        num_coord = target_coord.shape[1]
        target_coord_tokens = self.coord_to_tokens(target_coord) # Batch x num_coord x transformer_dim
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # Batch x (64 x 64) x 256
        visual_reference_embedding = visual_reference_embedding.flatten(2).permute(0, 2, 1)  # Batch x (64 x 64) x 256

        tokens = torch.cat((visual_reference_embedding, target_coord_tokens), dim=1)  # Batch x N_queries x transformer_dim, where N_queries = (num_crops x H x W) + num_coord

        for i in range(self.num_layers):
            '''tokens = self.transformer_cross_attention_layers[i](target=tokens, source=image_embedding, target_pos=None, source_mask=source_mask, source_pos=None)
            tokens = self.transformer_self_attention_layers[i](target=tokens, target_mask=target_mask, target_pos=None)
            tokens = self.transformer_ffn_layers[i](target=tokens)'''
            tokens = checkpoint(self.transformer_cross_attention_layers[i], tokens, image_embedding, source_mask)
            tokens = checkpoint(self.transformer_self_attention_layers[i], tokens, target_mask)
            tokens = checkpoint(self.transformer_ffn_layers[i], tokens)

        last_tokens = tokens[:, -num_coord:, :]  # Take the last tokens from the sequence
        final_output_coordinate = self.tokens_to_coord(last_tokens)  # transform it to a 4-dimensional space

        return final_output_coordinate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / self.max_epochs))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
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

        # Initialize the sequence with the first target coordinate
        output_coords = torch.zeros_like(target_coord)
        output_coords[:, 0, :] = target_coord[:, 0, :]

        torch.cuda.empty_cache()
        gc.collect()

        for t in range(1, target_coord.shape[1]):
            print(t)
            output_coord = self.forward(
                target_coord=output_coords[:, :t, :],
                image_embedding=image_embedding,
                visual_reference_embedding=visual_reference_embedding,
                source_mask=source_mask,
                target_mask=target_mask[:, :t + visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3], :t + visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3]]
            )
            output_coords[:, t, :] = output_coord[:, -1, :]
            del output_coord
            torch.cuda.empty_cache()
            gc.collect()

        pad = torch.tensor([-1, -1, -1, -1], dtype=torch.float32).to(self.device)
        valid_mask = (target_coord != pad).all(dim=-1)
        valid_output_coord = output_coords[valid_mask]
        valid_target_coord = target_coord[valid_mask]

        loss = self.loss_func(valid_output_coord, valid_target_coord)
        iou = self.calculate_iou(valid_output_coord, valid_target_coord)

        self.training_step_outputs.append({'loss': loss, 'iou': iou})

        return {'loss': loss}
    
    def on_train_epoch_end(self):
        avg_train_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        all_ious = torch.cat([x['iou'] for x in self.training_step_outputs])
        avg_train_iou = all_ious.mean()
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

        output_coords = torch.zeros_like(target_coord)
        output_coords[:, 0, :] = target_coord[:, 0, :]

        for t in range(1, target_coord.shape[1]):
            output_coord = self.forward(
                target_coord=output_coords[:, :t, :],
                image_embedding=image_embedding,
                visual_reference_embedding=visual_reference_embedding,
                source_mask=source_mask,
                target_mask=target_mask[:, :t + visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3], :t + visual_reference_embedding.shape[2] * visual_reference_embedding.shape[3]]
            )
            output_coords[:, t, :] = output_coord[:, -1, :]

            del output_coord
            torch.cuda.empty_cache()
            gc.collect

        pad = torch.tensor([-1, -1, -1, -1], dtype=torch.float32).to(self.device)
        valid_mask = (target_coord != pad).all(dim=-1)
        valid_output_coord = output_coords[valid_mask]
        valid_target_coord = target_coord[valid_mask]

        val_loss = self.loss_func(valid_output_coord, valid_target_coord)
        iou = self.calculate_iou(valid_output_coord, valid_target_coord)

        self.val_step_outputs.append({'loss': val_loss, 'iou': iou})

        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean()
        all_ious = torch.cat([x['iou'] for x in self.val_step_outputs])
        avg_val_iou = all_ious.mean()
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
        xA = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        yA = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        xB = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        yB = torch.min(pred_boxes[:, 3], true_boxes[:, 3])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

        boxAArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        boxBArea = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])

        unionArea = boxAArea + boxBArea - interArea

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
