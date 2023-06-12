import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


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

# -------- SASRec with Window ---------- #
def data_partition_window(fname, window_size=7, target_seq_percentage=0.9):
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
            train_seq = User[user][0:len(User[user])-2]  # remove last two items from train set
            target_split = int(target_seq_percentage * len(train_seq))  # split train set into input and target sequences

            input_seq, target_seq = train_seq[:target_split], train_seq[target_split:]  # split train set into input and target sequences

            for single_target in target_seq:
                temp_input = input_seq.copy() # get all items from input_seq
                temp_input.append(single_target) # add single target item to input_seq
                index = index + 1 
                user_train[index] = temp_input # add to train set
            
            user_valid[user] = [User[user][-2]] # add second to last item to valid set
            user_test[user] = [User[user][-1]] # add last item to test set
    
    usernum = index
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate_window(model, dataset, args, k_future_item=7, num_of_samples=500, at_k=10):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # Metrics accumulators
    NDCG = {1: 0.0, k_future_item: 0.0}
    Recall = {1: 0.0, k_future_item: 0.0}
    HitRate = {1: 0.0, k_future_item: 0.0}
    valid_user = 0.0

    # Sampling items for negative samples
    random_items = random.sample(range(1, itemnum + 1), num_of_samples)
    sample_idx = random_items
    users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k_future_item: continue

        # Preparing input sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # Ground truth items (next and k-th)
        ground_truth_idx = valid[u][:k_future_item]

        # Getting predictions
        process_idx = ground_truth_idx + sample_idx
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], process_idx]])[0]

        # Sorting items by score
        sorted_indices = torch.argsort(predictions)[::-1]

        # Evaluating for both target positions (next and k-th future interaction)
        for target_pos in [1, k_future_item]:
            target_item = ground_truth_idx[target_pos-1]

            # Getting the top at_k recommendations
            top_n_items = torch.tensor(process_idx)[sorted_indices][:at_k]

            # Computing Recall@at_k
            recall = int(target_item in top_n_items)

            # Computing NDCG@at_k
            dcg = 0.0
            for i, item in enumerate(top_n_items, 1):
                if item == target_item:
                    dcg += 1 / np.log2(i + 1)
            ideal_dcg = 1 / np.log2(2)  # Ideal DCG@at_k
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            
            # Computing Hit Rate@at_k
            hit_rate = int(target_item in top_n_items)

            # Accumulating the metrics
            Recall[target_pos] += recall
            NDCG[target_pos] += ndcg
            HitRate[target_pos] += hit_rate

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging
    for target_pos in [1, k_future_item]:
        Recall[target_pos] /= valid_user
        NDCG[target_pos] /= valid_user
        HitRate[target_pos] /= valid_user

    return Recall, NDCG, HitRate


# evaluate on valid set window
def evaluate_valid_window(model, dataset, args, k_future_item=7, num_of_samples=500, at_k=10):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # Metrics accumulators
    NDCG = {1: 0.0, k_future_item: 0.0}
    Recall = {1: 0.0, k_future_item: 0.0}
    HitRate = {1: 0.0, k_future_item: 0.0}
    valid_user = 0.0

    # Sampling items for negative samples
    random_items = random.sample(range(1, itemnum + 1), num_of_samples)
    sample_idx = random_items
    users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k_future_item: continue

        # Preparing input sequence
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # Ground truth items (next and k-th)
        ground_truth_idx = valid[u][:k_future_item]

        # Getting predictions
        process_idx = ground_truth_idx + sample_idx
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], process_idx]])[0]

        # Sorting items by score
        sorted_indices = torch.argsort(predictions)[::-1]

        # Evaluating for both target positions (next and k-th future interaction)
        for target_pos in [1, k_future_item]:
            target_item = ground_truth_idx[target_pos-1]

            # Getting the top at_k recommendations
            top_n_items = torch.tensor(process_idx)[sorted_indices][:at_k]

            # Computing Recall@at_k
            recall = int(target_item in top_n_items)

            # Computing NDCG@at_k
            dcg = 0.0
            for i, item in enumerate(top_n_items, 1):
                if item == target_item:
                    dcg += 1 / np.log2(i + 1)
            ideal_dcg = 1 / np.log2(2)  # Ideal DCG@at_k
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            
            # Computing Hit Rate@at_k
            hit_rate = int(target_item in top_n_items)

            # Accumulating the metrics
            Recall[target_pos] += recall
            NDCG[target_pos] += ndcg
            HitRate[target_pos] += hit_rate

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging
    for target_pos in [1, k_future_item]:
        Recall[target_pos] /= valid_user
        NDCG[target_pos] /= valid_user
        HitRate[target_pos] /= valid_user

    return Recall, NDCG, HitRate
