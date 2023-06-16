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

# -------- SASRec with window ---------- #
def data_partition_window(fname):
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
            seq_len = len(User[user]) - 2  # exclude the last two elements from the sequence
            valid_index = int(seq_len * 0.8)  # index that corresponds to approximately 80% of the sequence length
            test_index = int(seq_len * 0.9)

            # Only the input sequences without target sequences
            user_train[user] = User[user][:valid_index]
            user_valid[user] = User[user][valid_index:test_index]
            user_test[user] = User[user][test_index:]

    return [user_train, user_valid, user_test, usernum, itemnum]

# -------- SASRec with Window and Split feed---------- #
def data_partition_window_split(fname, target_seq_percentage=0.9):
    usernum = 0
    itemnum = 0
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

    usernum = index
    return [user_train, user_train_seq, user_valid, user_test, usernum, itemnum]

# Evaluate on test set with window
def evaluate_window(model, dataset, args, dataset_window, k=7):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    [_, train, valid, test, _, itemnum] = copy.deepcopy(dataset_window)

    NDCG = [0.0] * k
    HT = [0.0] * k
    valid_user = 0.0
    count = 0
    # Limit the number of users evaluated
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < k: continue
        count += 1
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u] + valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        for j in range(k):
            item_indices = [test[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            ranks = predictions.argsort().argsort()  # rank items by their scores
            rank = ranks[0].item()
            if rank < 10:
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging NDCG and Hit Rate for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    print('count: ', count)
    ndcg_avg = statistics.mean(NDCG)   
    ht_avg = statistics.mean(HT)
    return NDCG, HT, ndcg_avg, ht_avg


# evaluate on valid set window
def evaluate_valid_window(model, dataset, args, dataset_window, k=7):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    [_, train, valid, test, _, itemnum] = copy.deepcopy(dataset_window)

    NDCG = [0.0] * k
    HT = [0.0] * k
    valid_user = 0.0
    
    # Limit the number of users evaluated
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k: continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        for j in range(k):
            item_indices = [valid[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            ranks = predictions.argsort().argsort()  # rank items by their scores
            rank = ranks[0].item()
            if rank < 10:
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # Averaging NDCG and Hit Rate for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    ndcg_avg = statistics.mean(NDCG)   
    ht_avg = statistics.mean(HT)

    return NDCG, HT, ndcg_avg, ht_avg

def evaluate_window_new(model, dataset, args, dataset_window, k_future_pos=7, top_N=10):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    [_, train, valid, test, _, itemnum] = copy.deepcopy(dataset_window)

    NDCG = [0.0] * k_future_pos
    HT = [0.0] * k_future_pos
    SEQUENCE_SCORE = [0.0] * k_future_pos
    HT_ORDERED_SCORE = [0.0] * k_future_pos
    KENDALL_SCORE = [0.0] * k_future_pos

    weight_ht, weight_ordering = 0.5, 0.5
    valid_user = 0.0
    count = 0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

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

        true_ordering = []
        predicted_ordering = []

        for j in range(k_future_pos):
            item_indices = [valid[u][j]]
            true_ordering.append(test[u][j])
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            # Find the item with the highest prediction score
            top_predicted_index = max(enumerate(predictions), key=lambda x: x[1])[0]

            top_predicted_item = item_indices[top_predicted_index]
            
            # Append the top predicted item to the predicted_ordering list
            predicted_ordering.append(top_predicted_item)

            # Ranking and Score calculations
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()
            
            # Calculating the Sequence Score
            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(true_ordering, predicted_ordering)
        KENDALL_SCORE[-1] += tau

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        if count <= 4:
            print("True Order: ",true_ordering)
            print("Predicted Order" ,predicted_ordering)

    # Averaging NDCG, Hit Rate and Sequence Score for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    SEQUENCE_SCORE = [score / valid_user for score in SEQUENCE_SCORE]
    HT_ORDERED_SCORE = [weight_ht * HT[i] + weight_ordering * SEQUENCE_SCORE[i] for i in range(k_future_pos)]
    KENDALL_SCORE = [score / valid_user for score in KENDALL_SCORE]
    print('count: ', count)
    ndcg_avg = statistics.mean(NDCG)
    ht_avg = statistics.mean(HT)
    sequence_score_avg = statistics.mean(SEQUENCE_SCORE)
    ht_ordered_score_avg = statistics.mean(HT_ORDERED_SCORE)
    kendall_score_avg = statistics.mean(KENDALL_SCORE)

    return NDCG, HT, SEQUENCE_SCORE, HT_ORDERED_SCORE, KENDALL_SCORE, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, kendall_score_avg

def evaluate_valid_window_new(model, dataset, args, dataset_window, k_future_pos=7, top_N=10):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    [_, train, valid, test, _, itemnum] = copy.deepcopy(dataset_window)

    NDCG = [0.0] * k_future_pos
    HT = [0.0] * k_future_pos
    SEQUENCE_SCORE = [0.0] * k_future_pos
    HT_ORDERED_SCORE = [0.0] * k_future_pos
    KENDALL_SCORE = [0.0] * k_future_pos

    weight_ht, weight_ordering = 0.5, 0.5
    valid_user = 0.0
    count = 0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

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

        true_ordering = []
        predicted_ordering = []

        for j in range(k_future_pos):
            item_indices = [valid[u][j]]
            true_ordering.append(test[u][j])
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            # Find the item with the highest prediction score
            top_predicted_index = max(enumerate(predictions), key=lambda x: x[1])[0]

            top_predicted_item = item_indices[top_predicted_index]
            
            # Append the top predicted item to the predicted_ordering list
            predicted_ordering.append(top_predicted_item)

            # Ranking and Score calculations
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()
            
            # Calculating the Sequence Score
            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        # Calculating Kendall's Tau for the sequence
        tau, _ = kendalltau(true_ordering, predicted_ordering)
        KENDALL_SCORE[-1] += tau

        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

        if count <= 4:
            print("True Order: ",true_ordering)
            print("Predicted Order" ,predicted_ordering)

    # Averaging NDCG, Hit Rate and Sequence Score for each position
    NDCG = [score / valid_user for score in NDCG]
    HT = [score / valid_user for score in HT]
    SEQUENCE_SCORE = [score / valid_user for score in SEQUENCE_SCORE]
    HT_ORDERED_SCORE = [weight_ht * HT[i] + weight_ordering * SEQUENCE_SCORE[i] for i in range(k_future_pos)]
    KENDALL_SCORE = [score / valid_user for score in KENDALL_SCORE]
    print('count: ', count)
    ndcg_avg = statistics.mean(NDCG)
    ht_avg = statistics.mean(HT)
    sequence_score_avg = statistics.mean(SEQUENCE_SCORE)
    ht_ordered_score_avg = statistics.mean(HT_ORDERED_SCORE)
    kendall_score_avg = statistics.mean(KENDALL_SCORE)

    return NDCG, HT, SEQUENCE_SCORE, HT_ORDERED_SCORE, KENDALL_SCORE, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, kendall_score_avg