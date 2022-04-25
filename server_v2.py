# -*- coding: utf8 -*-
import os, sys, json, time, argparse
import requests
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from server_utils import preprocess_turns
from flask_cors import CORS

import torch
import asa_infer_e2e
import asa_model
from transformers import BertTokenizer

app = Flask(__name__)
CORS(app)
api = Api(app)

polarity_map = {-1:'negative', 0:'neutral', 1:'positive'}

def extract_sentiment(phrases, offset):
    span = []
    for i in range(len(phrases)):
        for j in range(i):
            phr = ''.join(phrases[j:i+1])
            if phr in ('很有名气', '特别喜欢'):
                span.append({'senti_span':[j+offset,i+offset],
                    'polarity':'positive', 'mentn_span':[0,2]})
    return span


class CASAParser(Resource):
    def get(self):
        print("got a get request")

    def put(self):
        print("got a put request")

    def post(self):
        print("getting a post request ....")
        dialog_turns = None
        for key in request.form.to_dict(flat=False):
            jo = json.loads(key)
            if "dialog_turns" in jo:
                dialog_turns = jo["dialog_turns"]
                dialog_turns = ['A: ' + x if i%2 == 0 else 'B: ' + x for i, x in enumerate(dialog_turns)]
                break

        if dialog_turns is None:
            return {}

        senti_res = self.parse(dialog_turns)

        response = {"senti_res_str": json.dumps(senti_res)}
        return response


    def parse(self, utts):
        # word segment
        utts_ids = []
        offsets = [0,]
        for i in range(len(utts)):
            utts_ids.append(tokenizer.encode(utts[i]))
            if i < len(utts) - 1:
                offsets.append(offsets[-1] + len(utts_ids[i]))

        # call CASA model
        dialogue = {'conv': utts_ids}
        sentiments, mentions = asa_infer_e2e.decode_dialogue(FLAGS, dialogue, sentiment_model, mention_model, tokenizer)

        senti_res = []
        for senti, mentn in zip(sentiments, mentions):
            stn = senti['turn_id']
            sst, sed = senti['span']
            senti_str = tokenizer.decode(utts_ids[stn][sst:sed+1])
            mtn = mentn['turn_id']
            mst, med = mentn['span']
            mentn_str = tokenizer.decode(utts_ids[mtn][mst:med+1])
            senti_res.append({'senti_str':senti_str, 'mentn_str':mentn_str})
        print(senti_res)
        return senti_res

        #senti_res = []
        #for senti, mentn in zip(sentiments, mentions):
        #    x = {}
        #    sst, sed = senti['span']
        #    sbase = offsets[senti['turn_id']]
        #    x['senti_span'] = [sst - 1 + sbase, sed - 1 + sbase] # -1 is for omitting the A: or B: in the beginning
        #    x['polarity'] = polarity_map[senti['senti']]
        #    mst, med = mentn['span']
        #    mbase = offsets[mentn['turn_id']]
        #    x['mentn_span'] = [mst - 1 + mbase, med - 1 + mbase]
        #    senti_res.append(x)

        #if len(senti_res) == 0:
        #    return [], []

        #turn_words = [] # a list of utterances as list
        #for turn in turns:
        #    turn_words.append(turn.strip().split(" "))
        #words, idx_mapping = preprocess_turns(turn_words)

        #for senti in senti_res:
        #    senti['senti_span'][0] = idx_mapping[senti['senti_span'][0]]
        #    senti['senti_span'][1] = idx_mapping[senti['senti_span'][1]]
        #    senti['mentn_span'][0] = idx_mapping[senti['mentn_span'][0]]
        #    senti['mentn_span'][1] = idx_mapping[senti['mentn_span'][1]]

        #res = []
        #for i, senti in enumerate(senti_res):
        #    senti_st, senti_ed = senti['senti_span']
        #    senti_str = ''.join(words[senti_st:senti_ed+1])
        #    mentn_st, mentn_ed = senti['mentn_span']
        #    mentn_str = ''.join(words[mentn_st:mentn_ed+1])
        #    args = [{'name': senti['polarity'], 'str': mentn_str}]
        #    jo = {"senti_name": senti_str, "args": args, "senti_st": senti_st, "senti_ed": senti_ed}
        #    res.append(jo)

        #return words, res

    def decode(self, segmented_test):
        pass

api.add_resource(CASAParser, '/casa')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Should be consistent with training')
    parser.add_argument('--cuda_device', type=str, required=True, help='GPU ids (e.g. "1" or "1,2")')
    parser.add_argument('--bert_version', type=str, required=True, help='BERT version (e.g. "bert-base-chinese"')
    parser.add_argument('--tok2word_strategy', type=str, required=True, help='Should be consistent with training, e.g. avg')
    parser.add_argument('--mention_model_path', type=str, required=True, help='The saved mention model')
    parser.add_argument('--sentiment_model_path', type=str, required=True, help='The saved sentiment model')
    FLAGS, unparsed = parser.parse_known_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}'.format(device, n_gpu))

    tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_version)

    print('Compiling model')
    mention_model = asa_model.BertAsaMe.from_pretrained(FLAGS.bert_version)
    mention_model.load_state_dict(torch.load(FLAGS.mention_model_path))
    mention_model.to(device)
    mention_model.eval()

    sentiment_model = asa_model.BertAsaSe.from_pretrained(FLAGS.bert_version)
    sentiment_model.load_state_dict(torch.load(FLAGS.sentiment_model_path))
    sentiment_model.to(device)
    sentiment_model.eval()
    print('Conversational Aspect Sentiment Analysis service is now available')
    app.run(host='0.0.0.0', port=2205)

    # def texsmart_api(text):
    #     obj = {"str": text}
    #     req_str = json.dumps(obj).encode()
    #     url = "https://texsmart.qq.com/api"
    #     r = requests.post(url, data=req_str)
    #     r.encoding = "utf-8"
    #     print(r.text)
    #
    # texsmart_api("她是很有名气的一位女歌手。")
    # texsmart_api("你 知道 她 是 哪年 出生 的 吗 ？")
