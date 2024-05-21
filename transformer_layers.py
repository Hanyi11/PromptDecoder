from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class CrossAttentionLayer(nn.Module):
    """
    Implements a cross-attention layer with optional pre-normalization.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - nhead (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - normalize_before (bool): Whether to apply normalization before the attention mechanism.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, normalize_before: bool = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Initializes parameters with Xavier uniform distribution for tensors with more than one dimension."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Optionally adds positional embeddings to the tensor.
        
        Args:
        - tensor (Tensor): Input tensor.
        - pos (Optional[Tensor]): Positional tensor to be added to the input tensor, if not None.
        
        Returns:
        - Tensor: Modified tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor, source: Tensor,
                     target_pos: Optional[Tensor] = None,
                     source_mask: Optional[Tensor] = None,
                     source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor used as key and value in attention.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        - source_mask (Optional[Tensor]): Optional mask for the source tensor.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source tensor.
        
        Returns:
        - Tensor: Output tensor after applying cross-attention and normalization.
        """
        target2 = self.multihead_attn(query=self.with_pos_embed(target, target_pos),
                                      key=self.with_pos_embed(source, source_pos),
                                      value=source, attn_mask=source_mask)[0]
        target = target + self.dropout(target2)
        target = self.norm(target)
        return target

    def forward_pre(self, target: Tensor, source: Tensor,
                    target_pos: Optional[Tensor] = None,
                    source_mask: Optional[Tensor] = None,
                    source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with pre normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor used as key and value in attention.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        - source_mask (Optional[Tensor]): Optional mask for the source tensor.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source tensor.
        
        Returns:
        - Tensor: Output tensor after applying normalization and cross-attention.
        """
        target2 = self.norm(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target2, target_pos),
                                      key=self.with_pos_embed(source, source_pos),
                                      value=source, attn_mask=source_mask)[0]
        target = target + self.dropout(target2)
        return target

    def forward(self, target: Tensor, source: Tensor,
                target_pos: Optional[Tensor] = None,
                source_mask: Optional[Tensor] = None,
                source_pos: Optional[Tensor] = None) -> Tensor:
        """
        Defines the forward pass with optional pre or post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - source (Tensor): Source sequence tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target.
        - source_mask (Optional[Tensor]): Optional mask for the source.
        - source_pos (Optional[Tensor]): Optional positional embeddings for the source.
        
        Returns:
        - Tensor: Output tensor after processing through the attention mechanism.
        """
        if self.normalize_before:
            return self.forward_pre(target, source, target_pos, source_mask, source_pos)
        return self.forward_post(target, source, target_pos, source_mask, source_pos)

class SelfAttentionLayer(nn.Module):
    """
    Implements a self-attention layer with optional pre-normalization.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - nhead (int): Number of attention heads.
    - dropout (float): Dropout rate.
    - normalize_before (bool): Whether to apply normalization before the attention mechanism.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, normalize_before: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """
        Initializes parameters with Xavier uniform distribution for tensors with more than one dimension.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Optionally adds positional embeddings to the tensor.
        
        Args:
        - tensor (Tensor): Input tensor.
        - pos (Optional[Tensor]): Positional tensor to be added to the input tensor, if not None.
        
        Returns:
        - Tensor: Modified tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor,
                     target_mask: Optional[Tensor] = None,
                     target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with post normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        
        Returns:
        - Tensor: Output tensor after applying self-attention and normalization.
        """
        q = k = self.with_pos_embed(target, target_pos)
        target2 = self.self_attn(q, k, value=target, attn_mask=target_mask)[0]
        target = target + self.dropout(target2)
        target = self.norm(target)
        return target

    def forward_pre(self, target: Tensor,
                    target_mask: Optional[Tensor] = None,
                    target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with pre normalization.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target tensor.
        
        Returns:
        - Tensor: Output tensor after applying normalization and self-attention.
        """
        target2 = self.norm(target)
        q = k = self.with_pos_embed(target2, target_pos)
        target2 = self.self_attn(q, k, value=target2, attn_mask=target_mask)[0]
        target = target + self.dropout(target2)
        return target

    def forward(self, target: Tensor,
                target_mask: Optional[Tensor] = None,
                target_pos: Optional[Tensor] = None) -> Tensor:
        """
        Defines the forward pass with optional pre or post normalization based on configuration.
        
        Args:
        - target (Tensor): Target sequence tensor.
        - target_mask (Optional[Tensor]): Optional mask for the target tensor.
        - target_pos (Optional[Tensor]): Optional positional embeddings for the target.
        
        Returns:
        - Tensor: Output tensor after processing through the self-attention mechanism.
        """
        if self.normalize_before:
            return self.forward_pre(target, target_mask, target_pos)
        return self.forward_post(target, target_mask, target_pos)

def _get_activation_fn(activation: str) -> callable:
    """
    Returns the activation function based on the string identifier.

    Args:
    - activation (str): Name of the activation function ('relu', 'gelu', or 'glu').

    Returns:
    - callable: The corresponding PyTorch activation function.

    Raises:
    - RuntimeError: If the activation function name is not recognized.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation should be 'relu', 'gelu', or 'glu', not {activation}.")

class FFNLayer(nn.Module):
    """
    Implements a feedforward neural network layer as used in transformers, with options for
    pre-normalization and various activations.
    
    Parameters:
    - d_model (int): Dimensionality of the model.
    - dim_feedforward (int): Dimensionality of the hidden layer.
    - dropout (float): Dropout rate.
    - activation (str): Type of activation function to use ('relu', 'gelu', 'glu').
    - normalize_before (bool): Whether to apply normalization before other operations.
    """
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 activation: str = "relu", normalize_before: bool = False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initializes parameters with Xavier uniform distribution for tensors with more than one dimension.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Optionally adds positional embeddings to the tensor.

        Args:
        - tensor (Tensor): Input tensor.
        - pos (Optional[Tensor]): Positional tensor to be added to the input tensor, if not None.

        Returns:
        - Tensor: Modified tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, target: Tensor) -> Tensor:
        """
        Forward pass with post-normalization.

        Args:
        - target (Tensor): Input tensor to the feedforward network.

        Returns:
        - Tensor: Output tensor after processing through the feedforward network and normalization.
        """
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target))))
        target = target + self.dropout(target2)
        target = self.norm(target)
        return target

    def forward_pre(self, target: Tensor) -> Tensor:
        """
        Forward pass with pre-normalization.

        Args:
        - target (Tensor): Input tensor to the feedforward network.

        Returns:
        - Tensor: Output tensor after normalization and processing through the feedforward network.
        """
        target2 = self.norm(target)
        target2 = self.linear2(self.dropout(self.activation(self.linear1(target2))))
        target = target + self.dropout(target2)
        return target

    def forward(self, target: Tensor) -> Tensor:
        """
        Defines the forward pass with optional pre or post-normalization based on configuration.

        Args:
        - target (Tensor): Input tensor to the feedforward network.

        Returns:
        - Tensor: Output tensor after processing through the feedforward network.
        """
        if self.normalize_before:
            return self.forward_pre(target)
        return self.forward_post(target)

class PositionalEncoding(nn.Module):
    """
    A class to inject some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    This allows the model to learn the importance of a token's position within the sequence.

    Attributes:
        d_model (int): The dimensionality of the input embeddings.
        max_len (int): The maximum length of the input sequences.
        encoding (nn.Embedding): The embedding layer to encode positional information.
    """

    def __init__(self, d_model: int = 256, max_len: int = 500) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int, optional): The dimensionality of the input embeddings. Defaults to 256.
            max_len (int, optional): The maximum length of the input sequences. Defaults to 500.
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The tensor with positional encodings added, same shape as input.
        """
        length = x.size(1)
        positions = torch.arange(length, dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), length)
        return self.encoding(positions)
