import numpy as np
from utils import evaluate_window_soa_test 
from model import SASRec
import argparse


def get_default_args():
    args = argparse.Namespace()
    args.dataset = 'your_dataset'
    args.train_dir = 'your_train_dir' 
    args.batch_size = 128
    args.lr = 0.001
    args.maxlen = 200
    args.hidden_units = 50
    args.num_blocks = 2
    args.num_epochs = 201
    args.num_heads = 1
    args.dropout_rate = 0.5
    args.l2_emb = 0.0
    args.device = 'cpu'
    args.inference_only = False
    args.state_dict_path = None
    args.window_size = 7
    args.window_eval = True
    args.window_eval_size = 7
    args.data_partition = None
    args.model_training = None
    args.optimizer = 'adam'
    args.loss_type = 'bce'
    args.strategy = 'default'
    args.masking = False
    args.mask_prob = 0.15
    args.uniform_ss = False
    args.temporal = False
    return args

def compute_sequence_score(true_order, predicted_order, window_size):
    SEQUENCE_SCORE = 0
    for j in range(len(true_order)):
        true_pos = true_order[j]
        predicted_pos = predicted_order[j]
        seq_score = (window_size - abs(true_pos - predicted_pos)) / window_size
        SEQUENCE_SCORE += seq_score
    return SEQUENCE_SCORE / len(true_order)


def test_sequence_score():
    # Test case 1: perfect order
    true_order = [1, 2, 3]
    predicted_order = [1, 2, 3]
    score = compute_sequence_score(true_order, predicted_order, len(true_order))
    assert abs(score - 1.0) < 1e-6, f"For true_order={true_order} and predicted_order={predicted_order}, expected score=1.0 but got score={score}"

    # Test case 2: completely reversed order
    true_order = [1, 2, 3]
    predicted_order = [3, 2, 1]
    score = compute_sequence_score(true_order, predicted_order, len(true_order))
    expected_score = 0.5555555555555555
    assert abs(score - expected_score) < 1e-6, f"For true_order={true_order} and predicted_order={predicted_order}, expected score={expected_score} but got score={score}"

    # Test case 3: one item out of place
    true_order = [1, 2, 3]
    predicted_order = [1, 3, 2]
    score = compute_sequence_score(true_order, predicted_order, len(true_order))
    expected_score = (1 + 2/3 + 2/3) / 3
    assert abs(score - expected_score) < 1e-6, f"For true_order={true_order} and predicted_order={predicted_order}, expected score={expected_score} but got score={score}"

    # Test case 4: all items in wrong positions
    true_order = [1, 2, 3]
    predicted_order = [2, 3, 1]
    score = compute_sequence_score(true_order, predicted_order, len(true_order))
    expected_score = (2/3 + 1/3 + 2/3) / 3
    assert abs(score - expected_score) < 1e-6, f"For true_order={true_order} and predicted_order={predicted_order}, expected score={expected_score} but got score={score}"

    print("All tests passed!")

def test_evaluate_window():

    mock_datasets = [
        {
            'train': {1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12], 4: [13, 14, 15, 16]},
            'valid': {1: [17], 2: [18], 3: [19], 4: [20]},
            'test': {1: [21, 22, 23], 2: [24, 25, 26, 27, 28, 29, 30], 3: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], 4: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]},
            'usernum': 4,
            'itemnum': 54
        },
    ]

    k_future_pos_values = [2, 3, 5]
    top_N_values = [5, 10]

    args = get_default_args()

    for dataset in mock_datasets:
        train = dataset['train']
        valid = dataset['valid']
        test = dataset['test']
        usernum = dataset['usernum']
        itemnum = dataset['itemnum']

        for k_future_pos in k_future_pos_values:
            for top_N in top_N_values:
                model = SASRec(usernum, itemnum, args) 

                # evaluate window function
                NDCG, HT, SEQUENCE_SCORE, HT_ORDERED_SCORE, ndcg_avg, ht_avg, sequence_score_avg, ht_ordered_score_avg, avg_kendall_tau = evaluate_window_soa_test(
                    model, [train, valid, test, usernum, itemnum], args, k_future_pos, top_N)

                # expected lengths in the outputs test
                assert len(NDCG) == k_future_pos
                assert len(HT) == k_future_pos
                assert len(SEQUENCE_SCORE) == k_future_pos
                assert len(HT_ORDERED_SCORE) == k_future_pos

                # expected averages are between 0 and 1 test
                assert 0 <= ndcg_avg <= 1
                assert 0 <= ht_avg <= 1
                assert 0 <= sequence_score_avg <= 1
                assert 0 <= ht_ordered_score_avg <= 1

                # kendall tau to be between -1 and 1 test
                assert -1 <= avg_kendall_tau <= 1

    print("All tests passed!")


test_evaluate_window()
test_sequence_score()
