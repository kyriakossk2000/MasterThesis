import math
import sys
import copy
import torch
import random
import statistics
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from scipy.stats import kendalltau

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

# sampler for batch generation for all action and dense all action
def random_neq_all(l, r, s, count):
    possible_numbers = list(set(range(l, r)) - set(s))
    np.random.shuffle(possible_numbers)
    return possible_numbers[:count]

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def sample_function_all(user_train_seq, train_target_seq, usernum, itemnum, batch_size, maxlen, result_queue, SEED, model_training, window_size):
    def sample():
        pos_samples = window_size
    
        user = np.random.randint(1, usernum + 1)
        while len(user_train_seq[user]) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)  # interaction sequence

        idx = maxlen - 1
        ts = set(user_train_seq[user] + train_target_seq[user])
        if model_training == 'all_action':
            neg_samples = 10
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
            pos = np.zeros([maxlen, pos_samples], dtype=np.int32)
            for i in reversed(user_train_seq[user]):
                seq[idx] = i 
                if idx == maxlen - 1:
                    # pad or truncate the train_target_seq[user] to make it of shape (7,)
                    target_samples = train_target_seq[user][:pos_samples]
                    target_samples += [0] * (pos_samples - len(target_samples))
                    pos[idx] = target_samples
                    neg[idx] = random_neq_all(1, itemnum + 1, ts, neg_samples)
                idx -= 1
                if idx == -1: break
        elif model_training == 'dense_all_action':
            neg_samples = 1
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
            for i in reversed(user_train_seq[user]):
                seq[idx] = i 
                random_target = random.sample(train_target_seq[user], 1)[0]
                pos[idx] = random_target
                neg[idx] = random_neq_all(1, itemnum + 1, ts, neg_samples)
                idx -= 1
                if idx == -1: break

        return user, seq, pos, neg

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

# All action and Dense all action sampler based on Pinnerformer 
class WarpSamplerAll(object):
    def __init__(self, user_input_seq, user_target_seq, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, model_training='all_action', window_size=7):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_all, args=(user_input_seq,
                                                        user_target_seq,
                                                        usernum,
                                                        itemnum,
                                                        batch_size,
                                                        maxlen,
                                                        self.result_queue,
                                                        np.random.randint(2e9),
                                                        model_training,
                                                        window_size
                                                        )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# -------- ORIGINAL SASRec ---------- #
# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    # create partitions
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# -------- SASRec with window ---------- #
def data_partition_window_baseline(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    # create partitions
    index = 0
    for user in User:
        nfeedback = len(User[user])
        
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            seq_len = len(User[user]) - 2  # exclude the last two elements from the sequence
            valid_index = int(seq_len * 0.8)  # index that corresponds to approximately 80% of the sequence length
            test_index = int(seq_len * 0.9)
            index += 1
            # Only the input sequences without target sequences
            user_train[user] = User[user][:valid_index]
            user_valid[user] = User[user][valid_index:test_index]
            user_test[user] = User[user][test_index:]

    return [user_train, user_valid, user_test, usernum, itemnum]

# -------- SASRec with Window and Split feed---------- #
def data_partition_window_independent(fname, target_seq_percentage=0.9):
    usernum = 0
    itemnum = 0
    train_samples = 0
    User = defaultdict(list)
    user_train = {}
    user_train_seq = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    # create partitions
    index = 0
    for user in User:
        nfeedback = len(User[user])
        
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            seq_len = len(User[user]) - 2  # exclude the last two elements from the sequence
            valid_index = int(seq_len * 0.8)  # index that corresponds to approximately 80% of the sequence length
            test_index = int(seq_len * 0.9)

            train_seq = User[user][:valid_index]
            valid_seq = User[user][valid_index:test_index]
            test_seq = User[user][test_index:]
            # splitting training sequence into input and target sequences based on the given target sequence percentage
            split_index = int(len(train_seq) * target_seq_percentage)
            input_seq = train_seq[:split_index]  
            target_seq = train_seq[split_index:]
            for single_target in target_seq:
                temp_input = input_seq.copy()
                temp_input.append(single_target)
                index += 1
                user_train[index] = temp_input

            user_train_seq[user] = train_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq

    train_samples = index
    return [user_train, user_train_seq, user_valid, user_test, usernum, itemnum, train_samples]

# -------- Partition with Window for Dense all action and All action ---------- #
def data_partition_window_all_action(fname, window_size=7, target_seq_percentage=0.9):
    train_start = target_seq_percentage
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_input = {}
    user_target = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    c = 0
    for user in User:
        nfeedback = len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            seq_len = len(User[user]) - 2
            valid_index = int(seq_len * 0.8)
            test_index = int(seq_len * 0.9)

            train_seq = User[user][:valid_index]
            valid_seq = User[user][valid_index:test_index]
            test_seq = User[user][test_index:]

            split_index = int(len(train_seq) * train_start)
            input_seq = train_seq[:split_index]
            target_seq = train_seq[split_index:]
            
            # Randomly sample or choose up to window size actions for target_seq
            if len(target_seq) < window_size:
                # Calculate number of elements needed to reach the window size
                num_needed = window_size - len(target_seq)
                # Sample the minimum between the number of elements needed and the length of target_seq
                additional_elements = random.sample(target_seq, min(num_needed, len(target_seq)))
                target_seq.extend(additional_elements)
            
            # Assign sequences to respective dictionaries
            user_input[user] = input_seq
            user_target[user] = target_seq
            user_train[user] = train_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq

    return [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum]

# Eval over window per step into the future 
def evaluate_window(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = [0.0] * k_future_pos
    HT = [0.0] * k_future_pos
    SEQUENCE_SCORE = [0.0] * k_future_pos
    HT_ORDERED_SCORE = [0.0] * k_future_pos

    weight_ht, weight_ordering = 0.5, 0.5
    valid_user = 0.0
    count = 0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    tau_scores = []
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < k_future_pos: continue
        count += 1
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u] + valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        pi_ri_pairs = []

        for j in range(k_future_pos):
            item_indices = [test[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            # Form (p_i, r_i) pairs
            pi_ri_pairs.append((j+1, rank+1))

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        # Unzip the pairs into two separate lists
        true_positions, predicted_rankings = zip(*pi_ri_pairs)
        if count < 5:
            print("True positions: ", true_positions)
            print("Predicted rankings: ", predicted_rankings)

        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(true_positions, predicted_rankings, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      

    # Averaging NDCG, Hit Rate, and Sequence Score for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    SEQUENCE_SCORE = [score / valid_user for score in SEQUENCE_SCORE]
    HT_ORDERED_SCORE = [weight_ht * HT[i] + weight_ordering * SEQUENCE_SCORE[i] for i in range(k_future_pos)]
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('count: ', count)
    ndcg_avg = statistics.mean(NDCG)
    ht_avg = statistics.mean(HT)
    sequence_score_avg = statistics.mean(SEQUENCE_SCORE)
    ht_ordered_score_avg = statistics.mean(HT_ORDERED_SCORE)

    return NDCG, HT, SEQUENCE_SCORE, HT_ORDERED_SCORE, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, avg_kendall_tau

def evaluate_valid_window(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = [0.0] * k_future_pos
    HT = [0.0] * k_future_pos
    SEQUENCE_SCORE = [0.0] * k_future_pos
    HT_ORDERED_SCORE = [0.0] * k_future_pos

    weight_ht, weight_ordering = 0.5, 0.5
    valid_user = 0.0
    count = 0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    tau_scores = []
    
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k_future_pos: continue
        count += 1
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        pi_ri_pairs = []

        for j in range(k_future_pos):
            item_indices = [valid[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            # Ranks
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            # Form (p_i, r_i) pairs
            pi_ri_pairs.append((j+1, rank+1))

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        # Unzip the pairs into two separate lists
        true_positions, predicted_rankings = zip(*pi_ri_pairs)

        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(true_positions, predicted_rankings, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging NDCG, Hit Rate, and Sequence Score for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    SEQUENCE_SCORE = [score / valid_user for score in SEQUENCE_SCORE]
    HT_ORDERED_SCORE = [weight_ht * HT[i] + weight_ordering * SEQUENCE_SCORE[i] for i in range(k_future_pos)]
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('count: ', count)
    ndcg_avg = statistics.mean(NDCG)
    ht_avg = statistics.mean(HT)
    sequence_score_avg = statistics.mean(SEQUENCE_SCORE)
    ht_ordered_score_avg = statistics.mean(HT_ORDERED_SCORE)

    return NDCG, HT, SEQUENCE_SCORE, HT_ORDERED_SCORE, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, avg_kendall_tau

# evaluates over all future sequences. K number of positives and draws 100 * neg samples
def evaluate_window_over_all(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    tau_scores = []

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < k_future_pos: continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u] + valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_indices = test[u][:k_future_pos]

        # Select k * 100 negative samples
        for _ in range(100 * k_future_pos):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_indices.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
        predictions = predictions[0]

        # Calculate statistics for the aggregated samples
        ranks = predictions.argsort().argsort()
        for j, rank in enumerate(ranks[:k_future_pos]):
            if rank < top_N:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(list(range(1, k_future_pos + 1)), ranks[:k_future_pos], variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
        
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging NDCG, Hit Rate
    NDCG /= valid_user
    HT /= valid_user
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('valid_user count: ', valid_user)

    return NDCG, HT, avg_kendall_tau

# evaluates over all future sequences. K number of positives and draws 100 * neg samples
def evaluate_window_over_all_valid(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    tau_scores = []

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k_future_pos: continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_indices = valid[u][:k_future_pos]

        # Select k * 100 negative samples
        for _ in range(100 * k_future_pos):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_indices.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
        predictions = predictions[0]

        # Calculate statistics for the aggregated samples
        ranks = predictions.argsort().argsort()
        for j, rank in enumerate(ranks[:k_future_pos]):
            if rank < top_N:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(list(range(1, k_future_pos + 1)), ranks[:k_future_pos], variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
        
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging NDCG, Hit Rate
    NDCG /= valid_user
    HT /= valid_user
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('valid_user count: ', valid_user)

    return NDCG, HT, avg_kendall_tau