import torch


class SampledSoftmaxLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_logits, neg_logits):

        num_neg_samples = neg_logits.size(1)
        
        # compute s(u_i, p_i) and s(u_i, n_j)
        s_pos = pos_logits / self.temperature
        s_neg = neg_logits / self.temperature

        # assuming uniform distribution and using a constant logQ
        log_q_pos = torch.log(torch.tensor(1.0 / num_neg_samples)).to(pos_logits.device)
        log_q_neg = torch.log(torch.tensor(1.0 / num_neg_samples)).to(neg_logits.device)
        
        numerator = torch.exp(s_pos - log_q_pos)
        
        denominator = numerator + torch.sum(torch.exp(s_neg - log_q_neg), dim=1, keepdim=True)
        
        # get sampled softmax loss for each example in the batch
        loss_per_example = -torch.log(numerator / denominator)
        
        loss = torch.mean(loss_per_example)
        
        return loss
    
class SampledSoftmaxLossOver(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(SampledSoftmaxLossOver, self).__init__()
        self.temperature = temperature

    def forward(self, pos_logits_list, neg_logits_list, neg_logQ_list):
        
        assert len(pos_logits_list) == len(neg_logits_list) == len(neg_logQ_list)
        
        total_loss = 0.0
        
        # iter over the window size
        for pos_logits, neg_logits, neg_logQ in zip(pos_logits_list, neg_logits_list, neg_logQ_list):

            # compute s(u_i, p_i) and s(u_i, n_j)
            s_pos = pos_logits / self.temperature
            s_neg = neg_logits / self.temperature
            
            corrected_s_pos = s_pos - neg_logQ  # neg_logQ is log(Q_i(p_i))
            corrected_s_neg = s_neg - neg_logQ  # neg_logQ is log(Q_i(n_j))
            
            numerator = torch.exp(corrected_s_pos)
            
            denominator = numerator + torch.sum(torch.exp(corrected_s_neg), dim=1, keepdim=True)
            
            loss_per_example = -torch.log(numerator / denominator)
            
            total_loss += torch.mean(loss_per_example)
        
        # norm by the window size
        return total_loss / len(pos_logits_list)
