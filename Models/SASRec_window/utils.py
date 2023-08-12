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
# TODO this method here could use uniform negative sampling 
def random_neq_all(l, r, s, count):
    possible_numbers = list(set(range(l, r)) - set(s))
    np.random.shuffle(possible_numbers)
    return possible_numbers[:count]

def random_neq_all_uniform(l, r, s, count):
    possible_numbers = list(set(range(l, r)) - set(s))
    np.random.shuffle(possible_numbers)
    return np.random.choice(possible_numbers, count, replace=True)

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

def sample_function_all(user_train_seq, train_target_seq, usernum, itemnum, batch_size, maxlen, result_queue, SEED, model_training, window_size, args):
    def sample():
        neg_samples = window_size
        user = np.random.randint(1, usernum + 1)
        while len(user_train_seq[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        seq = np.zeros([maxlen], dtype=np.int32)  # interaction sequence
        train_target_sampled = random.sample(train_target_seq[user], k=window_size) if len(train_target_seq[user]) > window_size else random.choices(train_target_seq[user], k=window_size)
        #train_target_sampled = random.sample(train_target_seq[user], k=16) if len(train_target_seq[user]) > 16 else random.choices(train_target_seq[user], k=16)

        idx = maxlen - 1
        ts = set(user_train_seq[user] + train_target_sampled)
        if model_training == 'all_action':
            pos_samples = window_size

            pos = np.zeros([maxlen, pos_samples], dtype=np.int32)
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
            for i in reversed(user_train_seq[user]):
                seq[idx] = i 
                if idx == maxlen - 1:
                    pos[idx] = train_target_sampled
                    if args.uniform_ss:
                        neg[idx] = random_neq_all_uniform(1, itemnum + 1, ts, neg_samples)
                    else:
                        for j in range(neg_samples):
                            neg[idx,j] = random_neq(1, itemnum + 1, ts)
                idx -= 1
                if idx == -1: break

        elif model_training == 'dense_all_action':
            pos_samples = 1
            pos = np.zeros([maxlen, pos_samples], dtype=np.int32)
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
            for i in reversed(user_train_seq[user]):
                seq[idx] = i 
                random_target = random.sample(train_target_sampled, 1)[0]
                pos[idx] = random_target
                neg[idx] = random_neq_all(1, itemnum + 1, ts, neg_samples)
                idx -= 1
                if idx == -1: break
        elif model_training == 'super_dense_all_action':
            pos_samples = window_size
            pos = np.zeros([maxlen, pos_samples], dtype=np.int32)
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
            for i in reversed(user_train_seq[user]):
                seq[idx] = i 
                pos[idx] = train_target_sampled
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
    def __init__(self, user_input_seq, user_target_seq, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, model_training='all_action', window_size=7, args=None):
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
                                                        window_size,
                                                        args
                                                        )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_function_combined(user_input_seq, user_target_seq, user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, model_training, window_size, args):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1 or len(user_input_seq[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)
        if args.data_partition == 'teacher_forcing':
            train_seq = random.choice(user_train[user])
            seq = np.zeros([maxlen], dtype=np.int32)
            pos = np.zeros([maxlen], dtype=np.int32)
            neg = np.zeros([maxlen], dtype=np.int32)
            nxt = train_seq[-1]
            idx = maxlen - 1
            ts = set(train_seq)
            for i in reversed(train_seq[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break
        else:
            # next item prediction
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

        # all action prediction
        neg_samples = window_size
        input_seq = user_input_seq[user]
        target_seq = user_target_seq[user]
        
        train_target_sampled = random.sample(target_seq, k=window_size) if len(target_seq) > window_size else random.choices(target_seq, k=window_size)
        ts_all = set(input_seq + train_target_sampled)

        
        pos_samples = window_size
        seq_all = np.zeros([maxlen], dtype=np.int32)
        pos_all = np.zeros([maxlen, pos_samples], dtype=np.int32)
        neg_all = np.zeros([maxlen, neg_samples], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(input_seq):
            seq_all[idx] = i 
            if idx == maxlen - 1:
                pos_all[idx] = train_target_sampled
                if args.uniform_ss:
                    neg_all[idx] = random_neq_all_uniform(1, itemnum + 1, ts_all, neg_samples)
                else:
                    for j in range(neg_samples):
                        neg_all[idx,j] = random_neq(1, itemnum + 1, ts_all)
            idx -= 1
            if idx == -1: break        

        return user, seq, pos, neg, seq_all, pos_all, neg_all

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSamplerCombined(object):
    def __init__(self, user_input_seq, user_target_seq, user_train, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, model_training='all_action', window_size=7, args=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_combined, args=(user_input_seq,
                                                        user_target_seq,
                                                        user_train, 
                                                        usernum,
                                                        itemnum,
                                                        batch_size,
                                                        maxlen,
                                                        self.result_queue,
                                                        np.random.randint(2e9),
                                                        model_training,
                                                        window_size,
                                                        args
                                                        )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def sample_function_all_temporal(user_train_seq, train_target_seq, usernum, itemnum, batch_size, maxlen, 
                        result_queue, SEED, model_training, window_size, args,
                        user_timestamp_seq=None, user_timestamp_target=None):
    def sample():
        neg_samples = window_size
        user = np.random.randint(1, usernum + 1)
        while len(user_train_seq[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        seq = np.zeros([maxlen], dtype=np.int32)  # interaction sequence
        time_seq = np.zeros([maxlen], dtype=np.float32)  # timestamp sequence
        train_target_sampled = random.sample(train_target_seq[user], k=window_size) if len(train_target_seq[user]) > window_size else random.choices(train_target_seq[user], k=window_size)
        item_to_time = dict(zip(train_target_seq[user], user_timestamp_target[user]))
        sampled_times = [item_to_time[item] for item in train_target_sampled]

        idx = maxlen - 1
        ts = set(user_train_seq[user] + train_target_sampled)
        if model_training == 'all_action':
            pos_samples = window_size

            pos = np.zeros([maxlen, pos_samples], dtype=np.int32)
            pos_time = np.zeros([maxlen, pos_samples], dtype=np.int32)
            neg = np.zeros([maxlen, neg_samples], dtype=np.int32)

            for i, t in reversed(list(zip(user_train_seq[user], user_timestamp_seq[user]))):
                seq[idx] = i
                time_seq[idx] = t
                if idx == maxlen - 1:
                    pos[idx] = train_target_sampled
                    pos_time[idx] = sampled_times
                    if args.uniform_ss:
                        neg[idx] = random_neq_all_uniform(1, itemnum + 1, ts, neg_samples)
                    else:
                        for j in range(neg_samples):
                            neg[idx,j] = random_neq(1, itemnum + 1, ts)
                idx -= 1
                if idx == -1: break
        
        return user, seq, pos, neg, time_seq, pos_time

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSamplerAllTemporal(object):
    def __init__(self, user_input_seq, user_target_seq, usernum, itemnum, batch_size=64, maxlen=10, 
                 n_workers=1, model_training='all_action', window_size=7, args=None, 
                 user_timestamp_input=None, user_timestamp_target=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_all_temporal, args=(user_input_seq,
                                                        user_target_seq,
                                                        usernum,
                                                        itemnum,
                                                        batch_size,
                                                        maxlen,
                                                        self.result_queue,
                                                        np.random.randint(2e9),
                                                        model_training,
                                                        window_size,
                                                        args,
                                                        user_timestamp_input,
                                                        user_timestamp_target
                                                        )))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def sample_function_all_temporal_combined(user_input_seq, user_target_seq, user_train, user_timestamp_input, user_timestamp_target, user_timestamp_train, usernum, itemnum, batch_size, maxlen, 
                        result_queue, SEED, model_training, window_size, args):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1 or len(user_input_seq[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)
        # next item prediction
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.float32)  # timestamp sequence
        pos = np.zeros([maxlen], dtype=np.int32)
        pos_time = np.zeros([maxlen], dtype=np.float32)  # pos timestamp sequence
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        nxt_time = user_timestamp_train[user][-1]
        idx = maxlen - 1
        ts = set(user_train[user])
        for i, t in reversed(list(zip(user_train[user][:-1], user_timestamp_train[user][:-1]))):
            seq[idx] = i
            time_seq[idx] = t
            pos[idx] = nxt
            pos_time[idx] = nxt_time
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            nxt_time = t
            idx -= 1
            if idx == -1: break

        # all action prediction
        neg_samples = window_size
        input_seq = user_input_seq[user]
        input_time_seq = user_timestamp_input[user]
        target_seq = user_target_seq[user]
        target_time_seq = user_timestamp_target[user]
        train_target_sampled = random.sample(target_seq, k=window_size) if len(target_seq) > window_size else random.choices(target_seq, k=window_size)
        train_time_sampled = [target_time_seq[target_seq.index(item)] for item in train_target_sampled]
        ts_all = set(input_seq + train_target_sampled)
        pos_samples = window_size
        seq_all = np.zeros([maxlen], dtype=np.int32)
        time_seq_all = np.zeros([maxlen], dtype=np.float32)  # timestamp sequence
        pos_all = np.zeros([maxlen, pos_samples], dtype=np.int32)
        pos_time_all = np.zeros([maxlen, pos_samples], dtype=np.float32)  # pos_all timestamp sequence
        neg_all = np.zeros([maxlen, neg_samples], dtype=np.int32)
        idx = maxlen - 1
        for i, t in reversed(list(zip(input_seq, input_time_seq))):
            seq_all[idx] = i 
            time_seq_all[idx] = t
            if idx == maxlen - 1:
                pos_all[idx] = train_target_sampled
                pos_time_all[idx] = train_time_sampled
                if args.uniform_ss:
                    neg_all[idx] = random_neq_all_uniform(1, itemnum + 1, ts_all, neg_samples)
                else:
                    for j in range(neg_samples):
                        neg_all[idx,j] = random_neq(1, itemnum + 1, ts_all)
            idx -= 1
            if idx == -1: break        

        return user, seq, pos, neg, time_seq, pos_time, seq_all, pos_all, neg_all, time_seq_all, pos_time_all


    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

class WarpSamplerAllTemporalCombined(object):
    def __init__(self, user_input, user_target, user_train, user_timestamp_input, user_timestamp_target, user_timestamp_train, usernum, itemnum, batch_size=64, maxlen=10, 
                 n_workers=1, model_training='all_action', window_size=7, args=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_all_temporal_combined, args=(user_input,
                                                        user_target,
                                                        user_train,
                                                        user_timestamp_input,
                                                        user_timestamp_target,
                                                        user_timestamp_train,
                                                        usernum,
                                                        itemnum,
                                                        batch_size,
                                                        maxlen,
                                                        self.result_queue,
                                                        np.random.randint(2e9),
                                                        model_training,
                                                        window_size,
                                                        args,
                                                        )))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def sample_function_rolling(user_input_seq, user_target_seq, usernum, itemnum, batch_size, maxlen, result_queue, SEED, window_size=7, args=None):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_input_seq[user]) <= 1:
            user = np.random.randint(1, usernum + 1)
        neg_samples = window_size

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen, window_size], dtype=np.int32)
        neg = np.zeros([maxlen, neg_samples], dtype=np.int32)
        idx = maxlen - 1

        ts = set(user_input_seq[user])
        for i, input_item in enumerate(reversed(user_input_seq[user])):
            seq[idx] = input_item
            pos[idx] = user_target_seq[user][i]
            if args.uniform_ss:
                neg[idx] = random_neq_all_uniform(1, itemnum + 1, ts, neg_samples)
            else:
                for j in range(neg_samples):
                    neg[idx,j] = random_neq(1, itemnum + 1, ts)
            idx -= 1
            if idx == -1: break
        
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSamplerRolling(object):
    def __init__(self, user_input_seq, user_target_seq, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, window_size=7, args=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_rolling, args=(user_input_seq,
                                                      user_target_seq,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      window_size,
                                                      args
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
        line_parts = line.rstrip().split()
        if len(line_parts) == 2:
            u, i = line_parts
        else:
            u, i, _, _ = line_parts
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


def data_partition_window_teacher_forcing(fname, target_seq_percentage=0.9):
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

            temp_input = input_seq.copy()
            for single_target in target_seq:
                temp_input.append(single_target)
                index += 1
                user_train[index] = temp_input.copy()

            user_train_seq[user] = train_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq

    train_samples = index
    return [user_train, user_train_seq, user_valid, user_test, usernum, itemnum, train_samples]


# -------- Partition with Window for Super, Dense all action and All action ---------- #
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
            
            if len(target_seq) < window_size:
                num_needed = window_size - len(target_seq)  # num of elements to reach window size 
                additional_elements = random.sample(target_seq, min(num_needed, len(target_seq))) # randomly sample or choose up to window size actions for target_seq
                target_seq.extend(additional_elements)
            
            user_input[user] = input_seq
            user_target[user] = target_seq
            user_train[user] = train_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq

    return [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum]


def data_partition_window_all_action_tf(fname, window_size=7, target_seq_percentage=0.9):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_input = {}
    user_target = {}
    user_train = defaultdict(list)
    user_train_seq = {}
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
    index = 0
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

            split_index = int(len(train_seq) * target_seq_percentage)
            input_seq = train_seq[:split_index]
            target_seq = train_seq[split_index:]
            
            if len(target_seq) < window_size:
                num_needed = window_size - len(target_seq)  
                additional_elements = random.sample(target_seq, min(num_needed, len(target_seq)))
                target_seq.extend(additional_elements)
            
            user_input[user] = input_seq
            user_target[user] = target_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq
            
            temp_input = input_seq.copy()
            for single_target in target_seq:
                temp_input.append(single_target)
                user_train[user].append(temp_input.copy())

            user_train_seq[user] = train_seq

    return [user_input, user_target, user_train, user_train_seq, user_valid, user_test, usernum, itemnum]


# -------- Partition with Window for Super, Dense all action and All action, but including temporal ---------- #
def data_partition_window_all_action_temporal(fname, window_size=7, target_seq_percentage=0.9):
    train_start = target_seq_percentage
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_input = {}
    user_target = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    # timestamp dictionary
    User_timestamp = defaultdict(list)
    user_timestamp_input = {}
    user_timestamp_target = {}
    user_timestamp_train = {}
    user_timestamp_valid = {}
    user_timestamp_test = {}

    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, _, t = line.rstrip().split()  # modify this line to split timestamp
        u = int(u)
        i = int(i)
        t = int(t)  # convert timestamp to integer or appropriate format
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        User_timestamp[u].append(t)  # add timestamp to dictionary

    c = 0
    for user in User:
        nfeedback = len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            # add timestamps
            user_timestamp_train[user] = User_timestamp[user]
            user_timestamp_valid[user] = []
            user_timestamp_test[user] = []
        else:
            seq_len = len(User[user]) - 2
            valid_index = int(seq_len * 0.8)
            test_index = int(seq_len * 0.9)

            train_seq = User[user][:valid_index]
            valid_seq = User[user][valid_index:test_index]
            test_seq = User[user][test_index:]

            # add timestamp sequences
            train_timestamp_seq = User_timestamp[user][:valid_index]
            valid_timestamp_seq = User_timestamp[user][valid_index:test_index]
            test_timestamp_seq = User_timestamp[user][test_index:]

            split_index = int(len(train_seq) * train_start)
            input_seq = train_seq[:split_index]
            target_seq = train_seq[split_index:]

            # add timestamp input and target sequences
            input_timestamp_seq = train_timestamp_seq[:split_index]
            target_timestamp_seq = train_timestamp_seq[split_index:]

            if len(target_seq) < window_size:
                num_needed = window_size - len(target_seq)  # num of elements to reach window size 
                additional_elements = random.sample(target_seq, min(num_needed, len(target_seq))) # randomly sample or choose up to window size actions for target_seq
                target_seq.extend(additional_elements)
                # repeat last timestamp for additional elements
                target_timestamp_seq.extend([target_timestamp_seq[-1]] * num_needed)

            user_input[user] = input_seq
            user_target[user] = target_seq
            user_train[user] = train_seq
            user_valid[user] = valid_seq
            user_test[user] = test_seq

            # add timestamps
            user_timestamp_input[user] = input_timestamp_seq
            user_timestamp_target[user] = target_timestamp_seq
            user_timestamp_train[user] = train_timestamp_seq
            user_timestamp_valid[user] = valid_timestamp_seq
            user_timestamp_test[user] = test_timestamp_seq

    return [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum, 
            user_timestamp_input, user_timestamp_target, user_timestamp_train, user_timestamp_valid, user_timestamp_test]


def data_partition_window_rolling(fname, window_size=2, target_seq_percentage=0.9):
    train_start = target_seq_percentage
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_input = {}
    user_target = {}
    user_train = {}
    user_valid = {}
    user_test = {}

    def create_rolling_window_sequences(seq, window_size):
        input_seqs = []
        target_seqs = []
        for i in range(len(seq)):
            input_seqs.append(seq[i])
            targets = seq[i+1:i+1+window_size]
            while len(targets) < window_size:
                targets.append(targets[-1] if targets else seq[i])
            target_seqs.append(targets)
        return input_seqs, target_seqs

    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    
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

            # create rolling windows for input and target sequences
            input_train, target_train = create_rolling_window_sequences(train_seq, window_size)
            input_valid, target_valid = create_rolling_window_sequences(valid_seq, window_size)
            input_test, target_test = create_rolling_window_sequences(test_seq, window_size)
            
            user_input[user] = input_train
            user_target[user] = target_train
            user_train[user] = input_train
            user_valid[user] = input_valid
            user_test[user] = input_test

    return [user_input, user_target, user_train, user_valid, user_test, usernum, itemnum]


# Eval over window per step into the future with time
def evaluate_window_time(model, dataset, args, k_future_pos=7, top_N=10):
    if args.temporal:
        [_, _, train, valid, test, usernum, itemnum, _, _, user_timestamp_train, user_timestamp_valid, _] = copy.deepcopy(dataset)
    else:
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

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
        time_seq = np.zeros([args.maxlen], dtype=np.float32)  # adjust the dtype as necessary

        idx = args.maxlen - 1
        for i, t in zip(reversed(train[u] + valid[u]), reversed(user_timestamp_train[u] + user_timestamp_valid[u])):
            seq[idx] = i
            time_seq[idx] = t  # assuming t is the timestamp
            idx -= 1
            if idx == -1: break

        rated = set(train[u] + valid[u])
        rated.add(0)

        model_predictions = []

        for j in range(k_future_pos):
            item_indices = [test[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices, [time_seq]]])
            predictions = predictions[0]
            
            model_predictions.append(item_indices[predictions.argsort()[0]])
            
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        if count < 5:
            print("True items: ", test[u][:k_future_pos])
            print("Predicted items: ", model_predictions)
        tau, _ = kendalltau(test[u][:k_future_pos], model_predictions, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      

    # averaging for each position
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

def evaluate_window_valid_time(model, dataset, args, k_future_pos=7, top_N=10):
    if args.temporal:
        [_, _, train, valid, test, usernum, itemnum, _, _, user_timestamp_train, user_timestamp_valid, _] = copy.deepcopy(dataset)
    else:
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

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
        time_seq = np.zeros([args.maxlen], dtype=np.float32)  # adjust the dtype as necessary

        idx = args.maxlen - 1
        for i, t in zip(reversed(train[u]), reversed(user_timestamp_train[u])):
            seq[idx] = i
            time_seq[idx] = t  # assuming t is the timestamp
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        model_predictions = []

        for j in range(k_future_pos):
            item_indices = [valid[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices, [time_seq]]])
            predictions = predictions[0]
            
            model_predictions.append(item_indices[predictions.argsort()[0]])
            
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1
        
        if count < 5:
            print("True items: ", valid[u][:k_future_pos])
            print("Predicted items: ", model_predictions)
        tau, _ = kendalltau(valid[u][:k_future_pos], model_predictions, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      

    # averaging for each position
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

# Eval over window per step into the future 
def evaluate_window(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action' or args.model_training == 'super_dense_all_action' or args.model_training == 'future_rolling' or args.model_training == 'combined':
        if args.data_partition == 'teacher_forcing' and args.model_training == 'combined':
            [_, _, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
        else:
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

        rated = set(train[u] + valid[u])
        rated.add(0)

        model_predictions = []

        for j in range(k_future_pos):
            item_indices = [test[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            model_predictions.append(item_indices[predictions.argsort()[0]])
            
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1

        if count < 5:
            print("True items: ", test[u][:k_future_pos])
            print("Predicted items: ", model_predictions)
        tau, _ = kendalltau(test[u][:k_future_pos], model_predictions, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      

    # averaging for each position
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
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action' or args.model_training == 'super_dense_all_action' or args.model_training == 'future_rolling' or args.model_training == 'combined':
        if args.data_partition == 'teacher_forcing' and args.model_training == 'combined':
            [_, _, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
        else:
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

        model_predictions = []

        for j in range(k_future_pos):
            item_indices = [valid[u][j]]
            for _ in range(99):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_indices.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
            predictions = predictions[0]
            
            model_predictions.append(item_indices[predictions.argsort()[0]])
            
            ranks = predictions.argsort().argsort()
            rank = ranks[0].item()

            if rank < top_N:
                seq_score = (k_future_pos - abs(j - rank)) / k_future_pos
                SEQUENCE_SCORE[j] += seq_score
                NDCG[j] += 1 / np.log2(rank + 2)
                HT[j] += 1

        if count < 5:
            print("True items: ", valid[u][:k_future_pos])
            print("Predicted items: ", model_predictions)
        tau, _ = kendalltau(valid[u][:k_future_pos], model_predictions, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      
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
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action' or args.model_training == 'super_dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    tau_scores = []
    neg_samples = 99  # Number of negative samples

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < k_future_pos:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u] + valid[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u] + valid[u])
        rated.add(0)

        # get k positive samples and some negative samples
        item_indices = test[u][:k_future_pos]
        while len(item_indices) < k_future_pos + neg_samples:
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_indices.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
        predictions = predictions[0]

        ranks = predictions.argsort().argsort()
        rank_positions = ranks[:k_future_pos].cpu().numpy()

        ndcg = sum(1 / np.log2(rank + 2) for rank in rank_positions if rank < top_N) / k_future_pos # Devide by k_future_pos to normalize
        ht = sum(1 for rank in rank_positions if rank < top_N) / k_future_pos 

        NDCG += ndcg
        HT += ht

        tau, _ = kendalltau(list(range(1, k_future_pos + 1)), [rank + 1 for rank in rank_positions], variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
        
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    # averaging the scores
    NDCG /= valid_user
    HT /= valid_user
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('\nValid users:', valid_user)

    return NDCG, HT, avg_kendall_tau

# evaluates over all future sequences. K number of positives and draws 100 * neg samples
def evaluate_window_over_all_valid(model, dataset, args, k_future_pos=7, top_N=10):
    if args.model_training == 'all_action' or args.model_training == 'dense_all_action' or args.model_training == 'super_dense_all_action':
        [_, _, train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    elif args.data_partition == None or args.data_partition == 'None':
        [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    else:
        [_, train, valid, test, usernum, itemnum, _] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    tau_scores = []
    neg_samples = 99  # Number of negative samples

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < k_future_pos:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        item_indices = valid[u][:k_future_pos]
        while len(item_indices) < k_future_pos + neg_samples:
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_indices.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_indices]])
        predictions = predictions[0]

        ranks = predictions.argsort().argsort()
        rank_positions = ranks[:k_future_pos].cpu().numpy()

        ndcg = sum(1 / np.log2(rank + 2) for rank in rank_positions if rank < top_N) / k_future_pos # divide by k_future_pos to normalize
        ht = sum(1 for rank in rank_positions if rank < top_N) / k_future_pos

        NDCG += ndcg
        HT += ht

        tau, _ = kendalltau(list(range(1, k_future_pos + 1)), [rank + 1 for rank in rank_positions], variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
        
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    NDCG /= valid_user
    HT /= valid_user
    avg_kendall_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0

    print('\nValid users:', valid_user)

    return NDCG, HT, avg_kendall_tau



def evaluate_window_soa_test(model, dataset, args, k_future_pos=7, top_N=10):
    [train, valid, test, usernum, itemnum] = dataset
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

        rated = set(train[u] + valid[u])
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
        
        true_positions, predicted_rankings = zip(*pi_ri_pairs)
        if count < 5:
            print("True positions: ", true_positions)
            print("Predicted rankings: ", predicted_rankings)

        tau, _ = kendalltau(true_positions, predicted_rankings, variant='b')
        if not math.isnan(tau):
            tau_scores.append(tau)
    
        valid_user += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
      

    # avveraging for each position
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
