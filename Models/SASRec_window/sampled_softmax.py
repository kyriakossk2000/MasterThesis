import torch


class SampledSoftmaxLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_logits, neg_logits):
        # Number of negative samples
        num_neg_samples = neg_logits.size(1)
        
        # Compute s(u_i, p_i) and s(u_i, n_j)
        s_pos = pos_logits / self.temperature
        s_neg = neg_logits / self.temperature

        # Compute logQ correction terms
        # Note: For simplicity assuming uniform distribution and using a constant logQ.
        log_q_pos = torch.log(torch.tensor(1.0 / num_neg_samples)).to(pos_logits.device)
        log_q_neg = torch.log(torch.tensor(1.0 / num_neg_samples)).to(neg_logits.device)
        
        numerator = torch.exp(s_pos - log_q_pos)
        
        denominator = numerator + torch.sum(torch.exp(s_neg - log_q_neg), dim=1, keepdim=True)
        
        # Compute the sampled softmax loss for each example in the batch
        loss_per_example = -torch.log(numerator / denominator)
        
        # Compute the mean loss over the batch
        loss = torch.mean(loss_per_example)
        
        return loss
