import os
import time
from sampled_softmax import SampledSoftmaxLoss, SampledSoftmaxLossOver
import torch
import argparse
from sam_optimizer.sam import SAM

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--window_size', default=7, type=int)                   # window size
parser.add_argument('--window_eval', default=False, type=str2bool)          # window evaluation or not
parser.add_argument('--window_eval_size', default=7, type=int)              # evaluate in the k position in the future 
parser.add_argument('--data_partition', default=None, type=str)             # type of data partition split -> independent, None (next item), teacher forcing, or autoregressive? 
parser.add_argument('--model_training', default=None, type=str)             # None is next item (SASRec), all action, or dense all action
parser.add_argument('--optimizer', default='adam', type=str)                # optimizer
parser.add_argument('--loss_type', default='bce', type=str)                 # loss function
parser.add_argument('--strategy', default='default', type=str)              # training strategy
parser.add_argument('--masking', default=False, type=str2bool)              # masking or not
parser.add_argument('--mask_prob', default=0.15, type=float)                # mask probability

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    print("Model training: ", args.model_training)
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action' or args.model_training == 'super_dense_all_action' or args.model_training == 'future_rolling':
        print("Training strategy: ", args.strategy)
        pass
    else:
        print("Data partition: ", args.data_partition)
    # load dataset
    if args.model_training == 'all_action':
        dataset = data_partition_window_all_action(args.dataset, window_size=args.window_size, target_seq_percentage=0.9)
        [user_input_seq, user_target_seq, user_train, user_valid, user_test, usernum, itemnum] = dataset
        training_samples = user_input_seq
        print("All action split:" + "\n" +"Number of training sequences in train set: " + str(len(user_input_seq.values())))
        count = 0
        for key, seq in user_input_seq.items():
            print(f"User: {key},Train Sequence: {seq}")
            print(f"Target Sequence for user {key}: ", user_target_seq.get(key, []))
            print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
        
        sampler = WarpSamplerAll(user_input_seq, user_target_seq, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, model_training=args.model_training, window_size=args.window_size, loss_type=args.loss_type)

    elif args.model_training == 'dense_all_action':
        dataset = data_partition_window_all_action(args.dataset, window_size=args.window_size, target_seq_percentage=0.9)
        [user_input_seq, user_target_seq, user_train, user_valid, user_test, usernum, itemnum] = dataset
        training_samples = user_input_seq
        print("Dense all action split:" + "\n" +"Number of training sequences in train set: " + str(len(user_input_seq.values())))
        count = 0
        for key, seq in user_input_seq.items():
            print(f"User: {key},Train Sequence: {seq}")
            print(f"Target Sequence for user {key}: ", user_target_seq.get(key, []))
            print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
        sampler = WarpSamplerAll(user_input_seq, user_target_seq, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, model_training=args.model_training, window_size=args.window_size, loss_type=args.loss_type)
    elif args.model_training == 'super_dense_all_action':
        dataset = data_partition_window_all_action(args.dataset, window_size=args.window_size, target_seq_percentage=0.9)
        [user_input_seq, user_target_seq, user_train, user_valid, user_test, usernum, itemnum] = dataset
        training_samples = user_input_seq
        print("Super dense all action split:" + "\n" +"Number of training sequences in train set: " + str(len(user_input_seq.values())))
        count = 0
        for key, seq in user_input_seq.items():
            print(f"User: {key},Train Sequence: {seq}")
            print(f"Target Sequence for user {key}: ", user_target_seq.get(key, []))
            print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
        sampler = WarpSamplerAll(user_input_seq, user_target_seq, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, model_training=args.model_training, window_size=args.window_size, loss_type=args.loss_type)
    
    elif args.model_training == 'future_rolling':
        dataset = data_partition_window_rolling(args.dataset, window_size=args.window_size, target_seq_percentage=0.9)
        [user_input_seq, user_target_seq, user_train, user_valid, user_test, usernum, itemnum] = dataset
        training_samples = user_input_seq
        print("Future window rolling:" + "\n" +"Number of training sequences in train set: " + str(len(user_input_seq.values())))
        count = 0
        for key, seq in user_input_seq.items():
            print(f"User: {key},Train Sequence: {seq}")
            print(f"Target Sequence for user {key}: ", user_target_seq.get(key, []))
            print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
            sampler = WarpSamplerRolling(user_input_seq, user_target_seq, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, window_size=args.window_size, loss_type=args.loss_type)
        
    else:  # SASRec next item
        if args.data_partition == 'independent':
            dataset = data_partition_window_independent(args.dataset, target_seq_percentage=0.9)
            [train_seq, user_train, user_valid, user_test, usernum, itemnum, train_samples] = dataset
            training_samples = train_seq
            print("Independent split:" + "\n" +"Number of training sequences in train set: " + str(len(train_seq.values())))
            count = 0
            for key, seq in train_seq.items():
                print(f"User: {key}, Sequence: {seq}")
                print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
                print(f"Test for user {key}: ", user_test.get(key, []))
                count += 1
                if count >= 3:  # Change this to print more or fewer sequences
                    break
            sampler = WarpSampler(train_seq, train_samples, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
        elif args.data_partition == 'teacher_forcing':
            dataset = data_partition_window_teacher_forcing(args.dataset, target_seq_percentage=0.9)
            [train_seq, user_train, user_valid, user_test, usernum, itemnum, train_samples] = dataset
            training_samples = train_seq
            print("Teacher forcing split:" + "\n" +"Number of training sequences in train set: " + str(len(train_seq.values())))
            count = 0
            for key, seq in train_seq.items():
                print(f"User: {key}, Sequence: {seq}")
                print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
                print(f"Test for user {key}: ", user_test.get(key, []))
                count += 1
                if count >= 3:  # Change this to print more or fewer sequences
                    break
            sampler = WarpSampler(train_seq, train_samples, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
        else:
            # Use baseline data partition
            dataset = data_partition_window_baseline(args.dataset)
            [user_train, user_valid, user_test, usernum, itemnum] = dataset
            training_samples = user_train
            print("Baseline next item split:" + "\n" + "Number of training sequences in train set: " + str(len(user_train.values())))
            count = 0
            for key, seq in user_train.items():
                print(f"User: {key}, Sequence: {seq}")
                print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
                print(f"Test for user {key}: ", user_test.get(key, []))
                count += 1
                if count >= 3:  # Change this to print more or fewer sequences
                    break
            sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    num_batch = len(training_samples) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in training_samples:
        cc += len(training_samples[u])
    print('average sequence length: %.2f' % (cc / len(training_samples)))
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    model = SASRec(usernum, itemnum, args).to(args.device) 
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()       
    
    if args.inference_only:
        model.eval()
        if not args.window_eval: 
            t_test = evaluate(model, dataset, args)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
        else:
            print('\n')
            print('Evaluating with window ' + str(args.window_eval_size))
            t_test = evaluate_window(model, dataset, args, k_future_pos=args.window_eval_size) 
            t_test_NDCG, t_test_HR, t_test_sequence_score, t_test_ht_ordered_score, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, t_test_kendall_avg = t_test
            #over_all_NDCG, over_all_HR, over_allKendall = evaluate_window_over_all(model, dataset, args, k_future_pos=args.window_eval_size)

            # print table headers
            # print('Evaluation statistics when evaluating over all k-steps into the future: ')
            # print("NDCG@10 Test Average: %.4f" % over_all_NDCG)
            # print("HR@10 Test Average: %.4f" % over_all_HR)
            # print("Kendall's Tau Average: %.4f" % over_allKendall)
            # print('\n')
            print('Evaluation statistics when evaluating for each k-step into the future: ')
            print("NDCG@10 Test Average: %.4f" % ndcg_avg)
            print("HR@10 Test Average: %.4f" % ht_avg)
            print("Sequence_Score@10 Test Average: %.4f" % sequence_score_avg)
            print("HT_Ordered@10 Test Average: %.4f" % ht_ordered_score_avg)
            print("Kendall's Tau Average: %.4f" % t_test_kendall_avg)
            print('{:<15}{:<10}{:<10}{:<10}{:<10}'.format("Position in future", "Test_NDCG", "Test_HR", "Test_Sequence_Score", "Test_HT_Ordered_Score"))
            
            for position in range(len(t_test_NDCG)):
                print('{:<15}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(
                    position + 1,
                    t_test_NDCG[position],
                    t_test_HR[position],
                    t_test_sequence_score[position],
                    t_test_ht_ordered_score[position]
                ))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    if args.loss_type == 'sampled_softmax':
        if args.model_training == 'all_action' or args.model_training == 'future_rolling':
            criterion = SampledSoftmaxLossOver()
        else:
            criterion = SampledSoftmaxLoss()
    elif args.loss_type == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss() 
    print('Loss type: ', args.loss_type)
    
    if args.optimizer == 'sam':
        base_optimizer = torch.optim.Adam
        sam_optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr)
    else:
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    if args.masking:
        mask_prob = args.mask_prob
    t0 = time.time()
    start_time = time.time()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            if args.masking:
                mask = np.random.choice([0, 1], size=(seq.shape[0], seq.shape[1]), p=[1-mask_prob, mask_prob])
                masked_seq = np.where(mask==1, 0, seq)
            
            if (args.model_training == 'future_rolling' or args.model_training == 'all_action') and args.loss_type == 'sampled_softmax':
                pos_logits, neg_logits, neg_logQ = model(u, seq, pos, neg)
            else:
                pos_logits, neg_logits = model(u, seq, pos, neg)
            if args.loss_type != 'ce_over' and not ((args.model_training == 'all_action' or args.model_training == 'future_rolling') and args.loss_type == 'sampled_softmax'):
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            if args.optimizer == 'sam':
                sam_optimizer.zero_grad()
                if args.model_training == 'all_action':
                    indices = -1
                    loss = criterion(pos_logits, pos_labels)
                    loss += criterion(neg_logits, neg_labels)
                else:
                    indices = np.where(pos != 0)
                    loss = criterion(pos_logits[indices], pos_labels[indices])
                    loss += criterion(neg_logits[indices], neg_labels[indices])
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward(retain_graph=True)
                sam_optimizer.first_step(zero_grad=True)
                # Second forward-backward pass
                if args.model_training == 'all_action':
                    loss = criterion(pos_logits, pos_labels)
                    loss += criterion(neg_logits, neg_labels)
                else:
                    loss = criterion(pos_logits[indices], pos_labels[indices])
                    loss += criterion(neg_logits[indices], neg_labels[indices])
                
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
                loss.backward()
                sam_optimizer.second_step(zero_grad=True)
                print('SAMM')
            else:
                adam_optimizer.zero_grad()
                if args.loss_type == 'sampled_softmax':
                    if args.model_training == 'all_action' or args.model_training == 'future_rolling':
                        loss = criterion(pos_logits, neg_logits, neg_logQ)
                    else:
                        loss = criterion(pos_logits, neg_logits) # compute the loss using SampledSoftmaxLoss
                        if args.masking:
                            mask_pos_logits, mask_neg_logits = model(u, masked_seq, seq, neg)
                            mask_loss = criterion(mask_pos_logits, mask_neg_logits)
                            loss += mask_loss
                elif args.loss_type == 'ce_over':
                    loss = 0
                    for i in range(args.window_size):
                        pos_labels, neg_labels = torch.ones(pos_logits[i].shape, device=args.device), torch.zeros(neg_logits[i].shape, device=args.device)
                        indices = np.where(pos[:,:,i] != 0)
                        logits = torch.cat((pos_logits[i][indices], neg_logits[i][indices]), dim=0)
                        labels = torch.cat((pos_labels[indices], neg_labels[indices]), dim=0)
                        for j in range(1,len(neg_logits)):
                            logits = torch.cat((logits, neg_logits[j][indices]), dim=0)
                            labels = torch.cat((labels, neg_labels[indices]), dim=0)
                        loss += criterion(logits, labels)
                    loss = loss.mean()  # avg over window size 
                    # pos_size = len(pos_logits)
                    # neg_size = len(neg_logits)
                    # max_size = max(pos_size, neg_size)
                    # for i in range(max_size):
                    #     if i < pos_size:
                    #         pos_labels = torch.ones(pos_logits[i].shape, device=args.device)
                    #         indices = np.where(pos[:,:,i] != 0)
                    #         pos_logits_i = pos_logits[i][indices]
                    #         pos_labels_i = pos_labels[indices]
                        
                    #     if i < neg_size:
                    #         neg_labels = torch.zeros(neg_logits[i].shape, device=args.device)
                    #         neg_logits_i = neg_logits[i][indices]
                    #         neg_labels_i = neg_labels[indices]

                    #     logits = torch.cat((pos_logits_i, neg_logits_i), dim=0)
                    #     labels = torch.cat((pos_labels_i, neg_labels_i), dim=0)

                    #     for j in range(1, neg_size):
                    #         if j != i:
                    #             logits = torch.cat((logits, neg_logits[j][indices]), dim=0)
                    #             labels = torch.cat((labels, neg_labels[indices]), dim=0)
                            
                    #     loss += criterion(logits, labels)

                    # loss = loss.mean()  # avg over window size 
                    
                    if args.masking:
                        mask_indices = np.where(mask == 1)
                        mask_pos_logits, mask_neg_logits = model(u, masked_seq, seq, neg[:,:,0])
                        mask_logits = torch.cat((mask_pos_logits[mask_indices], mask_neg_logits[mask_indices]), dim=0)
                        mask_pos_labels = torch.ones(mask_pos_logits.shape, device=args.device)
                        mask_neg_labels = torch.zeros(mask_neg_logits.shape, device=args.device)
                        mask_labels = torch.cat((mask_pos_labels[mask_indices], mask_neg_labels[mask_indices]), dim=0)
                        mask_loss = criterion(mask_logits, mask_labels)
                        loss += mask_loss
                else:
                    if args.model_training == 'all_action':
                        loss = criterion(pos_logits, pos_labels)
                        loss += criterion(neg_logits, neg_labels)
                    else:
                        indices = np.where(pos != 0)
                        loss = criterion(pos_logits[indices], pos_labels[indices]) 
                        loss += criterion(neg_logits[indices], neg_labels[indices])
                        if args.masking:
                            mask_indices = np.where(mask == 1)
                            mask_pos_logits, mask_neg_logits = model(u, masked_seq, seq, neg)
                            mask_pos_labels = torch.ones(mask_pos_logits.shape, device=args.device)
                            mask_neg_labels = torch.zeros(mask_neg_logits.shape, device=args.device)
                            mask_loss = criterion(mask_pos_logits[mask_indices], mask_pos_labels[mask_indices])
                            mask_loss += criterion(mask_neg_logits[mask_indices], mask_neg_labels[mask_indices])
                            loss += mask_loss

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
        
        if epoch % 50 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            if not args.window_eval:
                print('Evaluating Simple', end='')
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                        % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            else:
                print('\n')
                print('Evaluating with window ' + str(args.window_eval_size))
                t_test = evaluate_window(model, dataset, args, k_future_pos=args.window_eval_size)
                t_valid = evaluate_valid_window(model, dataset, args, k_future_pos=args.window_eval_size)
                t_test_NDCG, t_test_HR, t_test_sequence_score, t_test_ht_ordered_score, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, t_test_kendall_avg = t_test
                t_valid_NDCG, t_valid_HR, t_valid_sequence_score, t_valid_ht_ordered_score, valid_ndcg_avg, valid_ht_avg, valid_sequence_score_avg, valid_ht_ordered_score_avg, valid_kendall_avg = t_valid
                #over_all_NDCG, over_all_HR, over_allKendall = evaluate_window_over_all(model, dataset, args, k_future_pos=args.window_eval_size)
                #over_all_NDCG_valid, over_all_HR_valid, over_allKendall_valid = evaluate_window_over_all_valid(model, dataset, args, k_future_pos=args.window_eval_size)
                # print table headers
                # print('Evaluation statistics when evaluating over all k-steps into the future: ')
                # print('Test: ')
                # print("NDCG@10 Test Average: %.4f" % over_all_NDCG)
                # print("HR@10 Test Average: %.4f" % over_all_HR)
                # print("Kendall's Tau Average: %.4f" % over_allKendall)
                # print('Valid: ')
                # print("NDCG@10 Valid Average: %.4f" % over_all_NDCG_valid)
                # print("HR@10 Valid Average: %.4f" % over_all_HR_valid)
                # print("Kendall's Tau Average: %.4f" % over_allKendall_valid)
                # print('\n')
                print('Evaluation statistics when evaluating for each k-step into the future: ')
                print("Test" + '\n')
                print("NDCG@10 Test Average: %.4f" % ndcg_avg)
                print("HR@10 Test Average: %.4f" % ht_avg)
                print("Sequence_Score@10 Test Average: %.4f" % sequence_score_avg)
                print("HT_Ordered@10 Test Average: %.4f" % ht_ordered_score_avg)
                print("Kendall's Tau Test Average: %.4f" % t_test_kendall_avg)

                print("Valid" + '\n')
                print("NDCG@10 Valid Average: %.4f" % valid_ndcg_avg)
                print("HR@10 Valid Average: %.4f" % valid_ht_avg)
                print("Sequence_Score@10 Valid Average: %.4f" % valid_sequence_score_avg)
                print("HT_Ordered@10 Valid Average: %.4f" % valid_ht_ordered_score_avg)
                print("Kendall's Tau Valid Average: %.4f" % valid_kendall_avg)
                # print table headers
                print('{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format("Position in future", "Test_NDCG", "Test_HR", "Test_Sequence_Score", "Test_HT_Ordered_Score", "Valid_NDCG", "Valid_HR", "Valid_Sequence_Score", "Valid_HT_Ordered_Score"))
                for position in range(len(t_test_NDCG)):
                    print('{:<10d}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(
                        position + 1,
                        t_test_NDCG[position],
                        t_test_HR[position],
                        t_test_sequence_score[position],
                        t_test_ht_ordered_score[position],
                        t_valid_NDCG[position],
                        t_valid_HR[position],
                        t_valid_sequence_score[position],
                        t_valid_ht_ordered_score[position],
                    ))
                
    
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    end_time = time.time()
    total_time = end_time - start_time
    print("Execution time: ", total_time, " seconds")
    print("Done")
