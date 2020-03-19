
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
import asa_evaluator
import config_utils

FLAGS = None

def dev_eval(model, batches, log_file, verbose=0):
    print('Evaluating on devset')
    start = time.time()
    if FLAGS.task == 'sentiment':
        outputs = asa_evaluator.predict_sentiment(model, batches, verbose=verbose)
    else:
        outputs = asa_evaluator.predict_mention(model, batches)
    duration = time.time() - start
    print('Loss: %.2f, time: %.3f sec' % (outputs['loss'], duration))
    log_file.write('Loss: %.2f, time: %.3f sec\n' % (outputs['loss'], duration))
    if FLAGS.task == 'sentiment':
        p, r, f = outputs['score']
        f_un = outputs['score_un'][-1]
        accu = outputs['accu']
        print('F1: %.2f, (Precision: %.2f, Recall: %.2f), F1-un: %.2f, Accu: %.2f'%(100*f, 100*p, 100*r, 100*f_un, 100*accu))
        log_file.write('F1: %.2f, Precision: %.2f, Recall: %.2f, F1-un: %.2fAccu: %.2f\n'%(100*f, 100*p, 100*r, 100*f_un, 100*accu))
        log_file.flush()
        return f
    else:
        accu = outputs['score']
        print('Accuracy: %.2f' % (100*accu))
        log_file.write('Accuracy: %.2f\n' % (100*accu))
        log_file.flush()
        return accu


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
    train_features = asa_datastream.load_and_extract_features(FLAGS.train_path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    train_batches = asa_datastream.make_batch(train_features, FLAGS.task, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

    dev_features = asa_datastream.load_and_extract_features(FLAGS.dev_path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    dev_batches = asa_datastream.make_batch(dev_features, FLAGS.task, FLAGS.batch_size,
            is_sort=False, is_shuffle=False)

    test_features = asa_datastream.load_and_extract_features(FLAGS.test_path, tokenizer,
            FLAGS.tok2word_strategy, FLAGS.task)
    test_batches = asa_datastream.make_batch(test_features, FLAGS.task, FLAGS.batch_size,
            is_sort=False, is_shuffle=False)

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
    if os.path.exists(path_prefix + ".bert_model.bin"):
        print('!!Existing pretrained model. Loading the model...')
        model.load_state_dict(torch.load(path_prefix + ".bert_model.bin"))
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    if os.path.exists(path_prefix + ".bert_model.bin"):
        best_score = dev_eval(model, test_batches, log_file, verbose=1)
        print('Initial performance: {}'.format(best_score))
        sys.exit(0)
    else:
        best_score = 0.0

    update_steps = len(train_batches) * FLAGS.num_epochs
    if FLAGS.grad_accum_steps > 1:
        update_steps = update_steps // FLAGS.grad_accum_steps
    print('Starting the training loop, total epochs = {}, update steps = {}'.format(FLAGS.num_epochs, update_steps))

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(grouped_params,
            lr=FLAGS.learning_rate,
            warmup=FLAGS.warmup_proportion,
            t_total=update_steps)

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
            epoch_loss += loss.item()

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
        print('\nTraining loss: %s, time: %.3f sec' % (str(epoch_loss), duration))
        log_file.write('\nTraining loss: %s, time: %.3f sec\n' % (str(epoch_loss), duration))
        verbose = 0 if eid >= 5 else 0
        cur_score = dev_eval(model, dev_batches, log_file, verbose=verbose)
        if cur_score > best_score:
            print('Saving weights, score {} (prev_best) < {} (cur)'.format(best_score, cur_score))
            log_file.write('Saving weights, score {} (prev_best) < {} (cur)\n'.format(best_score, cur_score))
            best_score = cur_score
            save_model(model, path_prefix)
            FLAGS.best_score = best_score
            config_utils.save_config(FLAGS, path_prefix + ".config.json")
        print('-------------')
        log_file.write('-------------\n')
        #dev_eval(model, test_batches, log_file)
        #print('=============')
        #log_file.write('=============\n')
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

