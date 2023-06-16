import os
import time
import torch
import argparse

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
parser.add_argument('--window_split', default=False, type=str2bool)       # window prediction or not
parser.add_argument('--window_size', default=1, type=int)                # window size
parser.add_argument('--window_eval', default=False, type=str2bool)       # window evaluation or not
parser.add_argument('--window_eval_size', default=1, type=int)  # evaluate in the k position in the future 
parser.add_argument('--all_action', default=False, type=str2bool)  # all action prediction mimicking the Pinnerformer paper 

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # load dataset'
    # Use original data partition
    print("Window size: " + str(args.window_size))

    dataset = data_partition_window(args.dataset)
    user_id = 1
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    # Use window data partition
    dataset_window = data_partition_window_split(args.dataset, target_seq_percentage=0.9)
    if args.window_split:
        [user_train, train_seq, user_valid_window, user_test_window, usernum, itemnum_window] = dataset_window
        print("Number of training sequences in train set: " + str(len(user_train.values())))
        count = 0
        for key, seq in user_train.items():
            print(f"User: {key}, Sequence: {seq}")
            print(f"Valid for user {key}: ", user_valid_window.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test_window.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
        print("Itemnum: ", itemnum)
    else:
        print("Number of training sequences in train set: " + str(len(user_train.values())))
        count = 0
        for key, seq in user_train.items():
            print(f"User: {key}, Sequence: {seq}")
            print(f"Valid for user {key}: ", user_valid.get(key, []))  # Print validation and test data for a specific user
            print(f"Test for user {key}: ", user_test.get(key, []))
            count += 1
            if count >= 3:  # Change this to print more or fewer sequences
                break
        print("Itemnum: ", itemnum)
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
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
            print('Evaluating with window ' + str(args.window_eval_size) + '\n')
            t_test = evaluate_window_new(model, dataset, args, dataset_window) 
            t_test_NDCG, t_test_HR, t_test_sequence_score, t_test_ht_ordered_score, t_test_kendall, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, t_test_kendall_avg = t_test

            # print table headers
            print('\n')
            print("NDCG@10 Test Average: %.4f" % ndcg_avg)
            print("HR@10 Test Average: %.4f" % ht_avg)
            print("Sequence_Score@10 Test Average: %.4f" % sequence_score_avg)
            print("HT_Ordered@10 Test Average: %.4f" % ht_ordered_score_avg)
            print("Kendall's Tau Average: %.4f" % t_test_kendall_avg)
            print('{:<15}{:<10}{:<10}{:<10}{:<10}{:<10}'.format("Position in future", "Test_NDCG", "Test_HR", "Test_Sequence_Score", "Test_HT_Ordered_Score", "Test_Kendall"))
            
            for position in range(len(t_test_NDCG)):
                print('{:<15}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(
                    position + 1,
                    t_test_NDCG[position],
                    t_test_HR[position],
                    t_test_sequence_score[position],
                    t_test_ht_ordered_score[position],
                    t_test_kendall[position]
                ))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    start_time = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            if args.all_action:
                indices = -1
                loss = bce_criterion(pos_logits[:, indices], pos_labels[:, indices])
                loss = bce_criterion(neg_logits[:, indices], neg_labels[:, indices])
            else:
                indices = np.where(pos != 0) 
                loss = bce_criterion(pos_logits[indices], pos_labels[indices]) 
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 20 == 0:
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
                print('Evaluating with window ' + str(args.window_eval_size) + '\n')
                t_test = evaluate_window_new(model, dataset, args, dataset_window)
                t_valid = evaluate_valid_window_new(model, dataset, args, dataset_window)
                t_test_NDCG, t_test_HR, t_test_sequence_score, t_test_ht_ordered_score, t_test_kendall, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, t_test_kendall_avg = t_test
                t_valid_NDCG, t_valid_HR, t_valid_sequence_score, t_valid_ht_ordered_score, t_valid_kendall, valid_ndcg_avg, valid_ht_avg, valid_sequence_score_avg, valid_ht_ordered_score_avg, valid_kendall_avg = t_valid

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
                print('{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}'.format("Position in future", "Test_NDCG", "Test_HR", "Test_Sequence_Score", "Test_HT_Ordered_Score", "Test_Kendall Tau", "Valid_NDCG", "Valid_HR", "Valid_Sequence_Score", "Valid_HT_Ordered_Score", "Valid_Kendall Tau"))
                for position in range(len(t_test_NDCG)):
                    print('{:<10d}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}'.format(
                        position + 1,
                        t_test_NDCG[position],
                        t_test_HR[position],
                        t_test_sequence_score[position],
                        t_test_ht_ordered_score[position],
                        t_test_kendall[position],
                        t_valid_NDCG[position],
                        t_valid_HR[position],
                        t_valid_sequence_score[position],
                        t_valid_ht_ordered_score[position],
                        t_valid_kendall[position]
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
