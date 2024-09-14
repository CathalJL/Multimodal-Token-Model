import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
from jax import random
import optax
import numpy as np
#from tokenizers import Tokenizer
import os
import gc


"""
modules needed:
    token + pos embedding
    MHSA layer
    FFN definition
    layer norms
    resid connections

    output projections
    tokenizer

"""

# Layout 0-1
"""
Tokenization
self attention mechanism
ffn
normalization
resid
-----
data processing
-----
training block
incl: optimizer
loss function
ckpt saves 
evals ( loss + hellaswag+ some image gen)
-----
inference structure

"""



class LayerNorm(nn.Module): # change this later based on metas chameleon, for training stability
    epsilon: float = 1e-6  # A small constant to avoid division by zero

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / jnp.sqrt(variance + self.epsilon)
        gamma = self.param('gamma', nn.initializers.ones, (x.shape[-1],))
        beta = self.param('beta', nn.initializers.zeros, (x.shape[-1],))
        return gamma * normalized_x + beta
    

class MultiHeadSelfAttention(nn.Module):
    num_heads : int
    head_dim : int

    @nn.compact
    def __call__(self, x, mask=None):
        qkv = nn.Dense(self.num_heads * self.head_dim * 3)(x)
        qkv = qkv.reshape(x.shape[0], x.shape[1], self.num_heads, 3 * self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        scale = self.head_dim ** -0.5
        attn_weights = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * scale, axis=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.reshape(x.shape[0], x.shape[-1], -1)
        return nn.Dense(x.shape[-1])(attn_output)
    

# Testing the block ^
if __name__ == "__main__":
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    head_dim = 4

    # Create a dummy input tensor
    x = jnp.array(np.random.randn(batch_size, seq_length, d_model))

    # Initialize the multi-head self-attention layer
    multi_head_attention = MultiHeadSelfAttention(num_heads=num_heads, head_dim=head_dim)
    params = multi_head_attention.init(jax.random.PRNGKey(0), x)

    # Apply the multi-head self-attention
    output = multi_head_attention.apply(params, x)
    print("Output:\n", output)


class FeedForward(nn.Module):
    hidden_dim: int
    output_dim: int  # This will be the same as `d_model`

    @nn.compact
    def __call__(self, x):
        # First projection to a larger hidden dimension
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Project back to the original dimension (e.g., d_model)
        return nn.Dense(self.output_dim)(x)
    

class TransformerDecoderBlock(nn.Module):
    num_heads: int
    head_dim: int
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None):
        # NORM + MHSA
        x = nn.LayerNorm()(x)
        attn_output = MultiHeadSelfAttention(num_heads=self.num_heads, head_dim=self.head_dim)(x, mask)
        x = x + attn_output

        # NORM + FFN
        x = nn.LayerNorm()(x)
        ff_output = FeedForward(hidden_dim=self.hidden_dim, output_dim=x.shape[-1])(x)  # Project back to d_model
        x = x + ff_output

        return x
    

class PositionalEmbedding(nn.Module):
    max_length : int
    embedding_dim : int

    @nn.compact
    def __call__(self, x):
        position_indices = jnp.arange(self.max_length)
        positional_embeddings = self.param('positonal_embeddings', nn.initializers.normal(stddev=0.02), (self.max_length, self.embedding_dim))
        
        return positional_embeddings[position_indices[:x.shape[1]]]
    

class TokenEmbeddding(nn.Module):
    vocab_size : int
    embedding_dim : int

    @nn.compact
    def __call__(self, x):
        embedding_table = self.param('embedding_table', nn.initializers.normal(stddev=0.02), (self.vocab_size, self.embedding_dim))
        embeddings = embedding_table[x]

        return embeddings


class MultiModalModel(nn.Module):
    vocab_size: int
    max_length: int
    embedding_dim: int
    num_heads: int
    head_dim: int
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, combined_tokens):
        # Token embeddings
        token_embeddings = TokenEmbeddding(vocab_size=self.vocab_size, embedding_dim=self.embedding_dim)(combined_tokens)

        # Positional embeddings
        pos_emb = PositionalEmbedding(max_length=self.max_length, embedding_dim=self.embedding_dim)
        pos_embeddings = pos_emb(combined_tokens)

        # Add positional embeddings to token embeddings
        x = token_embeddings + pos_embeddings

        # Transformer decode block(s)
        for _ in range(self.num_layers):
            x = TransformerDecoderBlock(num_heads=self.num_heads, head_dim=self.head_dim, hidden_dim=self.hidden_dim)(x)

        # Split the combined output into text and image parts
        text_part = x[:, :seq_length, :]  # First part corresponds to text tokens
        image_part = x[:, seq_length:, :]  # Remaining part corresponds to image tokens

        # Output for text generation (vocab_size output to predict the next token)
        text_logits = nn.Dense(self.vocab_size)(text_part)

        # Output for image generation (assuming mapping embeddings back to image space)
        # Project down to a scalar to match the shape of image_labels
        image_outputs = nn.Dense(1)(image_part).squeeze(-1)

        return {
            'text_logits': text_logits,
            'image_outputs': image_outputs
        }

    


class TrainState(train_state.TrainState):
    apply_fn: callable = flax.struct.field()


def compute_loss(params, apply_fn, batch, mode="multimodel"):
    combined_tokens, text_labels, image_labels = batch
    outputs = apply_fn(params, combined_tokens)
    
    # Get the text and image logits
    text_logits = outputs['text_logits']
    image_outputs = outputs['image_outputs']

    # Cross entropy loss for text generation
    text_loss = optax.softmax_cross_entropy_with_integer_labels(text_logits, text_labels)
    text_loss = jnp.mean(text_loss)

    # MSE loss for image generation
    image_loss = jnp.mean(jnp.square(image_outputs - image_labels))

    total_loss = text_loss + image_loss
    return total_loss


def train_step(state, batch):
    """
    Performs a single training step (forward, loss, backprop, update)
    """
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, batch)
    
    grads = jax.grad(loss_fn)(state.params)

    state = state.apply_gradients(grads=grads)

    return state


def eval_step(params, model, batch):
    combined_tokens, text_labels, image_labels = batch
    outputs = model.apply(params, combined_tokens)

    # text accuracy (argmax over logits for token pred)
    text_logits = outputs['text_logits']
    text_preds = jnp.argmax(text_logits, axis=-1)
    text_accuracy = jnp.mean(text_preds == text_labels)

    # Image loss MSE
    image_outputs = outputs['image_outputs']
    image_loss = jnp.mean(jnp.square(image_outputs - image_labels))

    return text_accuracy, image_loss


def create_train_state(rng, model, learning_rate):
    """
    Initialize model parameters and optimizer.
    """
    # Create a dummy input sequence with integer token indices (e.g., all tokens set to 1)
    combined_dummy_tokens = jnp.ones((1, seq_length + image_seq_length), dtype=jnp.int32)  # Use integer tokens

    # Initialize model parameters with the combined dummy tokens
    params = model.init(rng, combined_dummy_tokens)

    # Set up the Adam optimizer
    tx = optax.adam(learning_rate)

    # Return the initialized training state
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_model(train_ds, val_ds, model, num_epochs=10, batch_size=32, learning_rate=0.001):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate)

    for epoch in range(num_epochs):
        # Train loopzoop
        for batch in train_ds: # at this point, train_ds has to be a batch dataset
            state = train_step(state, batch)

        # Eval
        total_text_acc = 0
        total_image_loss = 0
        for eval_batch in val_ds:
            text_acc, image_loss = eval_step(state.params, model, eval_batch)
            total_text_acc += text_acc
            total_image_loss += image_loss

        avg_text_acc = total_text_acc / len(val_ds) # change this monstrosity
        avg_image_loss = total_image_loss / len(val_ds)

        print(f"Epoch {epoch + 1}, Text Accuracy: {avg_text_acc:.4f}, Image Loss: {avg_image_loss:.4f}")




    # Testing the multimodal model
if __name__ == "__main__":
    batch_size = 8
    seq_length = 256
    image_seq_length = 256
    d_model = 512
    num_heads = 12
    head_dim = 64
    hidden_dim = 3072
    num_layers = 12
    vocab_size = 30000
    max_length = seq_length + image_seq_length # 512

    train_ds = [
    (jnp.concatenate([
        jnp.array(np.random.randint(0, vocab_size, (batch_size, seq_length))),  # Text tokens (batch_size, seq_length)
        jnp.array(np.random.randint(0, vocab_size, (batch_size, image_seq_length)))  # Image tokens as discrete integers
    ], axis=1),  # Combined (batch_size, seq_length + image_seq_length)
     jnp.array(np.random.randint(0, vocab_size, (batch_size, seq_length))),  # Text labels (for next token prediction)
     jnp.array(np.random.randint(0, vocab_size, (batch_size, image_seq_length)))  # Image labels as discrete tokens
    )  # Closing tuple for batch data
    for _ in range(100)
]

    val_ds = [
        (jnp.concatenate([
            jnp.array(np.random.randint(0, vocab_size, (batch_size, seq_length))),  # Text tokens
            jnp.array(np.random.randint(0, vocab_size, (batch_size, image_seq_length)))  # Image tokens as integers
        ], axis=1),
        jnp.array(np.random.randint(0, vocab_size, (batch_size, seq_length))),
        jnp.array(np.random.randint(0, vocab_size, (batch_size, image_seq_length)))
        )
        for _ in range(10)
]


    # Initialize the multimodal model
    multimodal_model = MultiModalModel(
        vocab_size=vocab_size,
        max_length=max_length,
        embedding_dim=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    # Start training
    train_model(train_ds, val_ds, multimodal_model)


