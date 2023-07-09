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
        self.strategy=args.strategy
        self.window_size=args.window_size
        self.loss_type=args.loss_type

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
            final_embedding = log_feats[:, -1, :]  # get last embedding element
            final_embedding_expanded = final_embedding.unsqueeze(1)
            
            if self.loss_type == 'ce_over':
                pos_logits_list = []
                neg_logits_list = []

                if self.strategy in ['autoregressive', 'teacher_forcing']:
                    seqs = torch.LongTensor(log_seqs)  # to a PyTorch tensor

                    for i in range(self.window_size):

                        log_feats = self.log2feats(seqs)
                        
                        pos_samples_embeddings = self.item_emb(torch.LongTensor(pos_seqs[:, :, i]).to(self.dev))
                        neg_samples_embeddings = self.item_emb(torch.LongTensor(neg_seqs[:, :, i]).to(self.dev))
                        pos_logits = (log_feats * pos_samples_embeddings).sum(dim=-1)
                        neg_logits = (log_feats * neg_samples_embeddings).sum(dim=-1)

                        pos_logits_list.append(pos_logits)
                        neg_logits_list.append(neg_logits)
                        
                        if self.strategy == 'autoregressive':
                            predicted_action = pos_logits.argmax(dim=-1)   # predictions
                            predicted_action = predicted_action.unsqueeze(1)
                        elif self.strategy == 'teacher_forcing':
                            predicted_action = torch.LongTensor(pos_seqs[:, :, i]).to(self.dev)  # actual positives
                            predicted_action = predicted_action[:,-1].unsqueeze(1)
                        seqs = seqs[:, 1:]  # remove the first element to maintain the embedding size
                        seqs = torch.cat([seqs, predicted_action], dim=1)

                    return pos_logits_list, neg_logits_list
                else:
                    if pos_seqs.ndim == 3:  # Handle the case where pos_seqs is 3D
                        for i in range(self.window_size):
                            pos_samples_embeddings = self.item_emb(torch.LongTensor(pos_seqs[:,:,i]).to(self.dev))
                            neg_samples_embeddings = self.item_emb(torch.LongTensor(neg_seqs[:,:,i]).to(self.dev))

                            pos_logits = (log_feats * pos_samples_embeddings).sum(dim=-1)
                            neg_logits = (log_feats * neg_samples_embeddings).sum(dim=-1)

                            pos_logits_list.append(pos_logits)
                            neg_logits_list.append(neg_logits)
                    elif pos_seqs.ndim == 2:
                        pos_samples_embeddings = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
                        neg_samples_embeddings = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
                        pos_logits = (log_feats * pos_samples_embeddings).sum(dim=-1)
                        neg_logits = (log_feats * neg_samples_embeddings).sum(dim=-1)
                        return pos_logits, neg_logits

                    return pos_logits_list, neg_logits_list
            
            elif self.loss_type == 'sampled_softmax':
                pos_logits_list = []
                neg_logits_list = []
                neg_logQ_list = []

                for i in range(self.window_size):
                    pos_embs = self.item_emb(torch.LongTensor(pos_seqs[:, :, i]).to(self.dev))
                    pos_logits_list.append((log_feats * pos_embs).sum(dim=-1))
                for j in range(neg_seqs.shape[2]):
                    neg_embs = self.item_emb(torch.LongTensor(neg_seqs[:, :, j]).to(self.dev))
                    neg_logQ = torch.zeros(neg_seqs[:, :, j].shape).to(self.dev)
                    for k in range(neg_seqs.shape[0]):
                        unique_negs, counts = torch.unique(torch.LongTensor(neg_seqs[:, :, j][k]), return_counts=True)  # times neg sample appers in batch
                        probs = counts.float() / neg_seqs.shape[0]  # probs of neg samples            
                        neg_logQ[k] = torch.log(probs).sum(dim=0)
                    neg_logQ_list.append(neg_logQ)
                    neg_logits_list.append((log_feats * neg_embs).sum(dim=-1))

                return pos_logits_list, neg_logits_list, neg_logQ_list
            else:
                pos_samples_embeddings = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))[:,-1,:]
                neg_samples_embeddings = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))[:,-1,:]
                
                pos_logits = (final_embedding_expanded * pos_samples_embeddings).sum(dim=-1)
                neg_logits = (final_embedding_expanded * neg_samples_embeddings).sum(dim=-1)
            

        elif self.model_training == 'dense_all_action' or self.model_training == 'super_dense_all_action':

            pos_seqs_as_tensor = torch.LongTensor(pos_seqs).to(self.dev)
            neg_seqs_as_tensor = torch.LongTensor(neg_seqs).to(self.dev)

            pos_sample_embeddings = self.item_emb(pos_seqs_as_tensor)
            neg_sample_embeddings = self.item_emb(neg_seqs_as_tensor)

            if self.model_training == 'dense_all_action':

                log_feats_expanded = log_feats.unsqueeze(2)

                pos_logits = (log_feats_expanded * pos_sample_embeddings).sum(dim=-1)                

                neg_logits = (log_feats_expanded * neg_sample_embeddings).sum(dim=-1)
            elif self.model_training == 'super_dense_all_action':
                log_feats_expanded = log_feats.unsqueeze(2)
                pos_logits = (log_feats_expanded * pos_sample_embeddings).sum(dim=-1)
                neg_logits = (log_feats_expanded * neg_sample_embeddings).sum(dim=-1) 
        elif self.model_training == 'future_rolling':

            pos_logits_list = []
            neg_logits_list = []
            neg_logQ_list = []

            if self.strategy in ['autoregressive', 'teacher_forcing']:
                if self.loss_type == 'ce_over':
                    seqs = torch.LongTensor(log_seqs).to(self.dev)  # o a PyTorch tensor

                    for i in range(self.window_size):

                        log_feats = self.log2feats(seqs)
                        
                        pos_samples_embeddings = self.item_emb(torch.LongTensor(pos_seqs[:, :, i]).to(self.dev))
                        neg_samples_embeddings = self.item_emb(torch.LongTensor(neg_seqs[:, :, i]).to(self.dev))
                        pos_logits = (log_feats * pos_samples_embeddings).sum(dim=-1)
                        neg_logits = (log_feats * neg_samples_embeddings).sum(dim=-1)

                        pos_logits_list.append(pos_logits)
                        neg_logits_list.append(neg_logits)
                        
                        if self.strategy == 'autoregressive':
                            predicted_action = pos_logits.argmax(dim=-1)   # predictions
                            predicted_action = predicted_action.unsqueeze(1)
                        elif self.strategy == 'teacher_forcing':
                            predicted_action = torch.tensor(pos_seqs[:, :, i]).to(self.dev).long()  # actual positives
                            predicted_action = predicted_action[:,-1].unsqueeze(1)
                        seqs = seqs[:, 1:]  # remove the first element to maintain the embedding size
                        seqs = torch.cat([seqs, predicted_action], dim=1)

                    return pos_logits_list, neg_logits_list
            else:
                for i in range(self.window_size):
                    pos_embs = self.item_emb(torch.LongTensor(pos_seqs[:, :, i]).to(self.dev))
                    pos_logits_list.append((log_feats * pos_embs).sum(dim=-1))

                if self.loss_type == 'ce_over':
                    for j in range(neg_seqs.shape[2]):    
                        neg_embs = self.item_emb(torch.LongTensor(neg_seqs[:, :, j]).to(self.dev))
                        neg_logits = (log_feats * neg_embs).sum(dim=-1)
                        neg_logits_list.append(neg_logits)
                    return pos_logits_list, neg_logits_list 
                elif self.loss_type == 'sampled_softmax':
                    for j in range(neg_seqs.shape[2]):
                        neg_embs = self.item_emb(torch.LongTensor(neg_seqs[:, :, j]).to(self.dev))
                        neg_logQ = torch.zeros(neg_seqs.shape[0], neg_seqs.shape[1]).to(self.dev)

                        for k in range(neg_seqs.shape[0]):
                            _, counts = torch.unique(torch.LongTensor(neg_seqs[:, :, j][k]), return_counts=True)  # times neg sample appers in batch
                            probs = counts.float() / neg_seqs.shape[0]  # probs of neg samples 
                            neg_logQ[k, :counts.size(0)] = torch.log(probs)

                        neg_logQ_list.append(neg_logQ)
                        neg_logits_list.append((log_feats * neg_embs).sum(dim=-1))
                    
                    return pos_logits_list, neg_logits_list, neg_logQ_list
                else:
                    for j in range(neg_seqs.shape[2]):
                        neg_embs = self.item_emb(torch.LongTensor(neg_seqs[:, :, i]).to(self.dev))
                        neg_logits = (log_feats * neg_embs).sum(dim=-1)
                        neg_logits_list.append(neg_logits)
                    pos_logits = torch.stack(pos_logits_list, dim=-1)
                    neg_logits = torch.stack(neg_logits_list, dim=-1)
                                   
        else:
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (U, I)
