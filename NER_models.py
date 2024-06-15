import torch
import utils
from tqdm import tqdm
import numpy as np
from torch import nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoding = torch.zeros(1024, hidden_size)
        self.encoding.requires_grad = False

        pos = torch.arange(0, 1024)
        pos = pos.float().unsqueeze(dim = 1)

        _2i = torch.arange(0, hidden_size, step = 2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        device = x.device

        return self.encoding[:seq_len, :].unsqueeze(0).to(device)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.d_head = self.hidden_size // self.num_heads
        self.scaling = self.d_head ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2).contiguous()

    def forward(self, query_states, key_value_states, attention_mask):
        attn_output = None

        batch_size, seq_len, _ = query_states.size() 
        kv_seq_len = key_value_states.size(1)

        query = self.q_proj(query_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)

        query = self._shape(query, seq_len, batch_size)
        key = self._shape(key, kv_seq_len, batch_size)
        value = self._shape(value, kv_seq_len, batch_size)

        query = query * self.scaling

        attention_scores = torch.matmul(query, key.transpose(-2, -1))

        if attention_mask is not None:
            attention_mask = utils._expand_mask(attention_mask, tgt_len=seq_len)
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        attn_output = torch.matmul(attention_probs, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output
        

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MultiHeadAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size)

        self.activation_fn = nn.ReLU()

        self.fc1 = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.fc2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, enc_self_mask):
        residual = hidden_states
        hidden_states = self.self_attn(
            query_states = hidden_states,
            key_value_states = hidden_states,
            attention_mask = enc_self_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config, embed_tokens, embed_positions):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, enc_ids, enc_mask):
        enc_hidden_states = None
        
        token_embeddings = self.embed_tokens(enc_ids)

        position_embeddings = self.embed_positions(enc_ids)

        enc_hidden_states = token_embeddings + position_embeddings

        enc_hidden_states = self.embedding_layer_norm(enc_hidden_states)

        for layer in self.layers:
            enc_hidden_states = layer(enc_hidden_states, enc_mask)

        return enc_hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_positions = PositionalEncoding(config.hidden_size)
        self.encoder = Encoder(config, self.embed_tokens, self.embed_positions)

    def forward(self, enc_ids, enc_mask = None):
        enc_hidden_states = self.encoder(enc_ids, enc_mask)

        return enc_hidden_states


class LSTMCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.hidden_size
        self.hidden_size = config.hidden_size

        self.linear_ih = nn.Linear(self.input_size, 4 * self.hidden_size)
        self.linear_hh = nn.Linear(self.hidden_size, 4 * self.hidden_size)

    def forward(self, input, hidden):
        hy, cy = None, None
        hx, cx = hidden

        gates = self.linear_ih(input) + self.linear_hh(hx)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)

        cy = f_gate * cx + i_gate * g_gate
        hy = o_gate * torch.tanh(cy)

        return hy, cy


class ModelForNER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer_encoder = TransformerEncoder(config)
        self.lstm = LSTMCell(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels
        self.use_lstm = config.use_lstm
    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                ):

        logits = None

        enc_hidden_states = self.transformer_encoder(input_ids, attention_mask)
        
        if self.use_lstm:
            batch_size, seq_len, _ = enc_hidden_states.size()
            hx = torch.zeros(batch_size, self.config.hidden_size, device=enc_hidden_states.device)
            cx = torch.zeros(batch_size, self.config.hidden_size, device=enc_hidden_states.device)
            outputs = []

            for t in range(seq_len):
                hx, cx = self.lstm(enc_hidden_states[:, t, :], (hx, cx))
                outputs.append(hx)

            enc_hidden_states = torch.stack(outputs, dim=1)

        enc_hidden_states = self.dropout(enc_hidden_states)
        logits = self.classifier(enc_hidden_states)

        return logits

    def train_model(self, train_loader, val_loader, optimizer, criterion, device, num_epochs=1):
        best_f1 = 0.0

        for epoch in range(num_epochs):
            print(f"########## EPOCH: {epoch} ##########")
            self.train()
            total_loss = 0

            for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="TRAIN"):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)

                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

            avg_loss_for_epoch = total_loss / len(train_loader)  

            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss_for_epoch}")

            metrics = self.validate_model(val_loader, device, is_test=False)

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(self.state_dict(), "best_ner_model.pth")
                print(f"New best model saved with accuracy: {best_f1:.4f}")

    def validate_model(self, loader, device, is_test=False):
        self.eval()
        valid_loss = 0.0
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for i, data in tqdm(enumerate(loader), total=len(loader), desc='VALIDATION'):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device)

                outputs = self(input_ids, attention_mask)

                all_predictions.append(np.argmax(outputs.cpu().numpy(), axis=2))
                all_labels.append(labels.cpu().numpy())

        metrics = utils.compute_metrics(all_predictions, all_labels, self.config.label_list)

        if is_test:        
            print(f'Test metrics: {metrics}')
        else:
            print(f"Validation metrics: {metrics}")

        return metrics