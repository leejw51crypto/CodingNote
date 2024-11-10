import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader

# 1. Multi-Head Self Attention Implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Create the query, key, and value projection layers
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        # Reshape the tensor for multi-head attention
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None):
        # Project inputs to query, key, and value
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

# 2. Transformer Block Implementation
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm1(attention_output + hidden_states)
        
        # Feed forward network
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(layer_output + attention_output)
        
        return layer_output

# 3. GPT Model Implementation
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights
        self.token_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, input_ids, attention_mask=None):
        # Get input embeddings
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        inputs_embeds = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Format attention mask for attention layers
        if attention_mask is not None:
            # Create attention mask that is broadcastable for multi-head attention
            # Convert from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert mask of 0s and 1s to mask of -10000.0 and 0.0
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
            
        # Get logits
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits

# 4. Training Configuration
class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        layer_norm_eps=1e-5,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps

# 5. Dataset Implementation
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                 max_length=max_length, return_tensors='pt')
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

# 6. Training Loop
def train(model, train_dataloader, optimizer, device, num_epochs):
    model.train()
    print(f"Starting training on device: {device}")
    print(f"Total batches per epoch: {len(train_dataloader)}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for batch in train_dataloader:
            batch_count += 1
            
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Print batch progress every 10 batches
            if batch_count % 10 == 0:
                avg_loss_so_far = total_loss / batch_count
                print(f"Batch {batch_count}/{len(train_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Avg Loss: {avg_loss_so_far:.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Total Loss: {total_loss:.4f}")

# 7. Example Usage
def main():
    # Initialize configuration
    config = GPTConfig(
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12
    )
    
    # Initialize model
    model = GPTModel(config)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # Create dummy data and dataloader
    dummy_texts = ["Your training text here"] * 100
    dataset = TextDataset(dummy_texts, tokenizer=tokenizer, max_length=512)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train the model
    train(model, train_dataloader, optimizer, device, num_epochs=3)

# Add this class before the main() function, right after the train() function
class SimpleTokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size

    def __call__(self, texts, truncation=True, padding='max_length', max_length=None, return_tensors='pt'):
        # Simple character-level tokenization
        batch_encoding = {'input_ids': [], 'attention_mask': []}
        
        for text in texts:
            # Convert characters to token ids (simple mapping: ord(char) % vocab_size)
            tokens = [ord(c) % self.vocab_size for c in text]
            
            # Truncate if necessary
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            # Pad if necessary
            attention_mask = [1] * len(tokens)
            if padding == 'max_length':
                padding_length = max_length - len(tokens)
                tokens.extend([0] * padding_length)
                attention_mask.extend([0] * padding_length)
            
            batch_encoding['input_ids'].append(tokens)
            batch_encoding['attention_mask'].append(attention_mask)
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            batch_encoding['input_ids'] = torch.tensor(batch_encoding['input_ids'])
            batch_encoding['attention_mask'] = torch.tensor(batch_encoding['attention_mask'])
            
        return batch_encoding

if __name__ == "__main__":
    main()