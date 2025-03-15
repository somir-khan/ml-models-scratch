import torch  # Import PyTorch library for tensor computations and neural networks
import torch.nn as nn  # Import neural network modules
import torch.optim as optim  # Import optimization algorithms
import torch.utils.data as data  # Import utilities for data handling
import math  # Import math library for mathematical functions
import copy  # Import copy library for deep copying objects

# Define the MultiHeadAttention class which implements multi-head self-attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model         # Model's total dimension
        self.num_heads = num_heads     # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension for each head (for keys, queries, and values)
        
        # Linear layers for transforming inputs into queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)  # Linear layer for query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Linear layer for key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Linear layer for value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Linear layer to combine the outputs of all heads
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores by computing dot product between queries and keys,
        # then scaling by the square root of d_k
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # If a mask is provided (e.g., to ignore padding tokens), apply it
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply the probabilities by the values to get the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Split the last dimension into (num_heads, d_k) and transpose so that the
        # attention mechanism can operate on the head dimension separately.
        batch_size, seq_length, d_model = x.size()
        # Reshape to (batch_size, seq_length, num_heads, d_k) then transpose to (batch_size, num_heads, seq_length, d_k)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Reverse the splitting operation by transposing and reshaping back to the original dimension.
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply the linear transformations and split heads for queries, keys, and values.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Compute scaled dot-product attention on the split representations.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine the multi-head outputs and pass them through a final linear layer.
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Define a PositionWiseFeedForward network that applies two linear transformations with a ReLU activation in between.
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # First linear layer: expands dimension to d_ff
        self.fc2 = nn.Linear(d_ff, d_model)  # Second linear layer: projects back to d_model
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        # Apply the first linear layer, ReLU activation, then the second linear layer.
        return self.fc2(self.relu(self.fc1(x)))
    
# Define PositionalEncoding to inject information about the position of tokens into the embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold the positional encodings for each position and dimension.
        pe = torch.zeros(max_seq_length, d_model)
        # Create a tensor for the positions (0, 1, 2, ..., max_seq_length-1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Compute the scaling factors for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register pe as a buffer to ensure it is not considered a model parameter
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to the input tensor x.
        return x + self.pe[:, :x.size(1)]

# Define the EncoderLayer which contains one multi-head self-attention layer followed by a feed-forward network.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Multi-head self-attention layer
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # Position-wise feed-forward network
        self.norm1 = nn.LayerNorm(d_model)  # Layer normalization after attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer normalization after feed-forward network
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        
    def forward(self, x, mask):
        # Apply self-attention to the input x with the provided mask.
        attn_output = self.self_attn(x, x, x, mask)
        # Apply dropout and add the residual connection followed by layer normalization.
        x = self.norm1(x + self.dropout(attn_output))
        # Apply the feed-forward network.
        ff_output = self.feed_forward(x)
        # Again, apply dropout and add the residual connection followed by layer normalization.
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Define the DecoderLayer which has a masked self-attention, cross-attention (encoder-decoder attention),
# and a feed-forward network.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Self-attention for target sequence
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # Cross-attention with encoder output
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)  # Feed-forward network
        self.norm1 = nn.LayerNorm(d_model)  # Layer normalization after self-attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer normalization after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # Layer normalization after feed-forward network
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Apply masked self-attention on the target sequence x.
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Apply cross-attention using the encoder output.
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # Pass through the feed-forward network.
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Define the overall Transformer model that includes encoder and decoder stacks.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        # Embedding layers for source and target vocabularies.
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Positional encoding layer to add position information to embeddings.
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Create a list (stack) of encoder layers.
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # Create a list (stack) of decoder layers.
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final fully connected layer that maps the output to the target vocabulary size.
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Generate source mask: True where src token is not 0 (assuming 0 is the pad token)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # Generate target mask for padding tokens
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        # Create a "no-peak" mask to ensure each position only attends to earlier positions (for autoregressive decoding)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # Create masks for the source and target sequences.
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # Get embeddings for the source and target and add positional encodings.
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Pass the source embeddings through each encoder layer.
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Pass the target embeddings through each decoder layer along with the encoder outputs.
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Compute the final output by applying a linear transformation.
        output = self.fc(dec_output)
        return output

# Hyperparameters for the model (can be tuned or changed based on your dataset)
src_vocab_size = 5000  # Size of the source vocabulary; update based on your dataset
tgt_vocab_size = 5000  # Size of the target vocabulary; update based on your dataset
d_model = 512          # Dimension of embeddings and hidden layers
num_heads = 8          # Number of attention heads
num_layers = 6         # Number of encoder and decoder layers
d_ff = 2048            # Dimension of the feed-forward layer
max_seq_length = 100   # Maximum sequence length; update as per your data
dropout = 0.1          # Dropout rate for regularization

# Instantiate the Transformer model with the specified parameters.
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data for training (dummy data for demonstration)
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # Source data with shape (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # Target data with shape (batch_size, seq_length)

# Define the loss function. Here CrossEntropyLoss is used and padding tokens (0) are ignored.
criterion = nn.CrossEntropyLoss(ignore_index=0)
# Define the optimizer. Adam optimizer is used with custom learning rate and betas.
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()  # Set the model to training mode

# Training loop for 100 epochs (update number of epochs as needed)
for epoch in range(100):
    optimizer.zero_grad()  # Reset gradients for the optimizer
    # Forward pass: Pass source data and target data (excluding last token) through the model
    output = transformer(src_data, tgt_data[:, :-1])
    # Compute loss by comparing model output and target data (offset by one for teacher forcing)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()  # Backpropagation to compute gradients
    optimizer.step()  # Update model parameters based on gradients
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")  # Print training progress

transformer.eval()  # Set the model to evaluation mode

# Generate random sample data for validation (dummy data for demonstration)
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

with torch.no_grad():
    # Evaluate the model on validation data without gradient calculations
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    # Compute validation loss similar to training loss
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")
