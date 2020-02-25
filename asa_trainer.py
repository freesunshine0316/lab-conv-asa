
import os, sys, json, codecs
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn
from tqdm import tqdm, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import asa_model
import asa_datastream
import config_utils

FLAGS = None


def dev_evaluate(model, evalset, device, log_file):
    print('Evaluating on dataset with data_type: {}'.format(data_type))
    model.eval()
    N = 0
    loss = 0.0
    start = time.time()
    # output and calculate performance
    total_loss = dev_loss['total_loss']
    duration = time.time()-start
    print('Loss: %.2f, time: %.3f sec' % (total_loss, duration))
    log_file.write('Loss: %.2f, time: %.3f sec\n' % (total_loss, duration))
    det_pr, det_rc, det_f1 = calc_f1(n_out=dev_counts['detection'][1],
            n_ref=dev_counts['detection'][2], n_both=dev_counts['detection'][0])
    #print('Detection F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*det_f1, 100*det_pr, 100*det_rc))
    log_file.write('Detection F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*det_f1, 100*det_pr, 100*det_rc))
    cur_result = {'data_type':data_type, 'loss':total_loss, 'detection_f1':det_f1}
    if data_type == 'recovery':
        rec_pr, rec_rc, rec_f1 = calc_f1(n_out=dev_counts['recovery'][1],
                n_ref=dev_counts['recovery'][2], n_both=dev_counts['recovery'][0])
        print('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*rec_f1, 100*rec_pr, 100*rec_rc))
        log_file.write('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*rec_f1, 100*rec_pr, 100*rec_rc))
        cur_result['key_f1'] = rec_f1
    else:
        res_pr, res_rc, res_f1 = calc_f1(n_out=dev_counts['resolution'][1],
                n_ref=dev_counts['resolution'][2], n_both=dev_counts['resolution'][0])
        print('Resolution F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*res_f1, 100*res_pr, 100*res_rc))
        log_file.write('Resolution F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*res_f1, 100*res_pr, 100*res_rc))
        cur_result['key_f1'] = res_f1
        resnp_pr, resnp_rc, resnp_f1 = calc_f1(n_out=dev_counts['resolution_nps'][1],
                n_ref=dev_counts['resolution_nps'][2], n_both=dev_counts['resolution_nps'][0])
        print('Resolution NP F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*resnp_f1, 100*resnp_pr, 100*resnp_rc))
        log_file.write('Resolution NP F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*resnp_f1, 100*resnp_pr, 100*resnp_rc))
        cur_result['resolution_np_f1'] = resnp_f1
    if len(development_sets) > 1:
        print('+++++')
        log_file.write('+++++\n')
    log_file.flush()
    evaluate_results.append(cur_result)
    model.train()
    return dev_eval_results


def main():
    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/ASA.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(str(FLAGS)))
    log_file.flush()
    config_utils.save_config(FLAGS, path_prefix + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))
    log_file.write('device: {}, n_gpu: {}, grad_accum_steps: {}\n'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if 'bert' in FLAGS.pretrained_path:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    # load data and make_batches
    print('Loading data and making batches')
    train_features = asa_datastream.load_and_extract_features(path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    train_batches = asa_datastream.make_batch(features, FLAGS.task, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

    dev_features = asa_datastream.load_and_extract_features(path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    dev_batches = asa_datastream.make_batch(features, FLAGS.task, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

    test_features = asa_datastream.load_and_extract_features(path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    test_batches = asa_datastream.make_batch(features, FLAGS.task, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

    print("Num training examples = {}".format(len(train_features)))
    print("Num training batches = {}".format(len(train_batches)))
    print("Data option: is_shuffle {}, is_sort {}, is_batch_mix {}".format(FLAGS.is_shuffle,
        FLAGS.is_sort, FLAGS.is_batch_mix))

    # create model
    print('Compiling model')
    if FLAGS.task == 'mention':
        model = asa_model.BertAsaMe.from_pretrained(FLAGS.pretrained_path)
    elif FLAGS.task == 'sentiment':
        model = asa_model.BertAsaSe.from_pretrained(FLAGS.pretrained_path)
    else:
        assert False, 'Unsupported task: ' + FLAGS.task
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    print('Starting the training loop, total epochs = {}'.format(FLAGS.num_epochs))

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(grouped_params,
            lr=FLAGS.learning_rate,
            warmup=FLAGS.warmup_proportion,
            t_total=train_steps)

    best_score = 0.0
    finished_steps, finished_epochs = 0, 0
    train_batch_ids = list(range(0, len(train_batches)))
    model.train()
    for eid in range(0, FLAGS.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        print('Current epoch takes {} steps'.format(len(train_batch_ids)))
        if FLAGS.is_batch_mix:
            random.shuffle(train_batch_ids)
        for bid in train_batch_ids:
            ori_batch = train_batches[bid]
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                    for k, v in ori_batch.items()}

            loss = model(batch)['loss']
            train_loss += loss.item()

            if n_gpu > 1:
                loss = loss.mean()
            if FLAGS.grad_accum_steps > 1:
                loss = loss / FLAGS.grad_accum_steps
            loss.backward() # just calculate gradient

            finished_steps += 1
            if finished_steps % 100 == 0:
                print('{} '.format(finished_steps), end="")
                sys.stdout.flush()

            if finished_steps % FLAGS.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        duration = time.time() - epoch_start
        print('\nTraining loss: %s, time: %.3f sec' % (str(train_loss), duration))
        log_file.write('\nTraining loss: %s, time: %.3f sec\n' % (str(train_loss), duration))
        cur_f1 = dev_eval(model, dev_batches, device, log_file):
        if cur_f1 > best_f1:
            print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
            log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
            best_f1 = cur_f1
            save_model(model, path_prefix)
            FLAGS.best_f1 = best_f1
            config_utils.save_config(FLAGS, path_prefix + ".config.json")
        print('-------------')
        log_file.write('-------------\n')
        dev_eval(model, test_batches, device, log_file)
        print('=============')
        log_file.write('=============\n')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finished_epochs += 1


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_bin_path = path_prefix + ".bert_model.bin"
    model_config_path = path_prefix + ".bert_config.json"

    torch.save(model_to_save.state_dict(), model_bin_path)
    model_to_save.config.to_json_file(model_config_path)


def check_config(FLAGS):
    assert type(FLAGS.grad_accum_steps) == int and FLAGS.grad_accum_steps >= 1
    assert hasattr(FLAGS, "cuda_device")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.config_path is not None:
        print('Loading hyperparameters from ' + FLAGS.config_path)
        FLAGS = config_utils.load_config(FLAGS.config_path)
    check_config(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    main()
