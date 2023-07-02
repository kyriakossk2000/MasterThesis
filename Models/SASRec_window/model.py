import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.model_training = args.model_training
        self.training_strategy=args.training_strategy

        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        
        if self.model_training == 'all_action':
            final_embedding = log_feats[:, -1, :]  # get last embedding el

            # Convert sequences to tensors and move them to the desired device
            pos_seqs_tensor = torch.LongTensor(pos_seqs).to(self.dev)
            neg_seqs_tensor = torch.LongTensor(neg_seqs).to(self.dev)
            
            # Get embeddings of the positive and negative samples (last elements)
            pos_samples_embeddings = self.item_emb(pos_seqs_tensor)[:, -1, :]
            neg_samples_embeddings = self.item_emb(neg_seqs_tensor)[:, -1, :]
            
            # Unsqueeze final_embedding for matrix multiplication
            final_embedding_expanded = final_embedding.unsqueeze(1)
            
            # Calculate the logits
            pos_logits = (final_embedding_expanded * pos_samples_embeddings).sum(dim=-1)
            neg_logits = (final_embedding_expanded * neg_samples_embeddings).sum(dim=-1)

        elif self.model_training == 'dense_all_action' or self.model_training == 'super_dense_all_action':
            # Convert positive and negative sequences to tensors and move them to the desired device
            pos_seqs_as_tensor = torch.LongTensor(pos_seqs).to(self.dev)
            neg_seqs_as_tensor = torch.LongTensor(neg_seqs).to(self.dev)

            # Get embeddings of the positive and negative samples
            pos_sample_embeddings = self.item_emb(pos_seqs_as_tensor)
            neg_sample_embeddings = self.item_emb(neg_seqs_as_tensor)

            if self.model_training == 'dense_all_action':
                # Expand the log_feats tensor dimension for matrix multiplication with neg_sample_embeddings
                log_feats_expanded = log_feats.unsqueeze(2)
                # Calculate the positive logits
                pos_logits = (log_feats_expanded * pos_sample_embeddings).sum(dim=-1)                
                # Calculate the negative logits
                neg_logits = (log_feats_expanded * neg_sample_embeddings).sum(dim=-1)
            elif self.model_training == 'super_dense_all_action':
                log_feats_expanded = log_feats.unsqueeze(2)
                pos_logits = (log_feats_expanded * pos_sample_embeddings).sum(dim=-1)
                neg_logits = (log_feats_expanded * neg_sample_embeddings).sum(dim=-1)            
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
            pos_logits = []
            neg_logits = []

            # Loop over sequence length
            for t in range(log_feats.shape[1]): # <-- Modification here
                pos_logits_t = (log_feats[:, t, :] * pos_embs[:, t, :]).sum(dim=-1) # <-- Modification here
                neg_logits_t = (log_feats[:, t, :] * neg_embs[:, t, :]).sum(dim=-1) # <-- Modification here
                
                pos_logits.append(pos_logits_t)
                neg_logits.append(neg_logits_t)

            pos_logits = torch.stack(pos_logits, dim=1)
            neg_logits = torch.stack(neg_logits, dim=1)
            # kame comment ta poupano os jame pou en ta list creation je uncomment to poukato 
            # pos_logits = (log_feats * pos_embs).sum(dim=-1)
            # neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
