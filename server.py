# -*- coding: utf8 -*-
import os, sys, json, time
import requests
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from server_utils import preprocess_turns
from flask_cors import CORS

import asa_infer_e2e

app = Flask(__name__)
CORS(app)
api = Api(app)

senti_map = {-1:'negative', 0:'neutral', 1:'positive'}

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
    def texsmart_api(self, text):
        obj = {"str": text}
        req_str = json.dumps(obj).encode()
        url = "https://texsmart.qq.com/api"
        r = requests.post(url, data=req_str)
        r.encoding = "utf-8"
        jo = json.loads(r.text)
        phrases_list = []
        tags_list = []
        for sub_jo in jo["res_list"]:
            phrases = []
            tags = []
            phrase_list = sub_jo["phrase_list"]
            for i in range(len(phrase_list)):
                phrases.append(phrase_list[i]["str"])
                tags.append(phrase_list[i]["tag"])
            phrases_list.append(phrases)
            tags_list.append(tags)
        return phrases_list, tags_list

    def get(self):
        print("got a get request")

    def post(self):
        print("getting a post request ....")
        dialog_list = None
        for key in request.form.to_dict(flat=False):
            jo = json.loads(key)
            if "dialog_list" in jo:
                dialog_list = jo["dialog_list"]
                break

        if dialog_list is None:
            return {}

        phrase_list, result = self.parse(dialog_list)
        print(result)

        response = {"senti_str": json.dumps(result), "phrase_list": phrase_list}
        return response


    def parse(self, input_json):
        # get info
        utts = []
        for i in range(1, 9, 1):
            key = "str{}".format(i)
            if key in input_json:
                utt = input_json[key].replace(" ", "").strip()
                if len(utt) == 0:
                    break
                else:
                    utts.append(utt)

        # word segment
        utts_as_words, _ = self.texsmart_api(utts)
        offsets = [0,]
        for i in range(len(utts_as_words)-1):
            offsets.append(offsets[-1] + len(utts_as_words[i]))

        # call CASA model
        utts_as_words = [['A:',] + x if i%2 == 0 else ['B:',] + x for i, x in enumerate(utts_as_words)]
        dialogue = {'conv': utts_as_words}
        sentiments, mentions = asa_infer_e2e.decode_dialogue(args, dialogue, sentiment_model, mention_model, tokenizer)

        senti_res = []
        for senti, mentn in zip(sentiments, mentions):
            x = {}
            sst, sed = senti['span']
            sbase = offsets[senti['turn_id']]
            x['senti_span'] = [sst - 1 + sbase, sed - 1 + sbase] # -1 is for omitting the A: or B: in the beginning
            x['polarity'] = senti['senti']
            mst, med = mentn['span']
            mbase = offsets[mentn['turn_id']]
            x['mentn_span'] = [mst - 1 + mbaes, med - 1 + mbase]
            senti_res.append(x)

        if len(senti_res) == 0:
            return []

        # don't know if that's necessary
        input = "<SEP>".join(utts)
        if input.endswith("<SEP>"):
            turns = input.split("<SEP>")[:-1]  # the last part is empty
        else:
            turns = input.split("<SEP>")

        turn_words = [] # a list of utterances as list
        for turn in turns:
            turn_words.append(turn.strip().split(" "))
        words, idx_mapping = preprocess_turns(turn_words)

        for senti in senti_res:
            senti['senti_span'][0] = idx_mapping[senti['senti_span'][0]]
            senti['senti_span'][1] = idx_mapping[senti['senti_span'][1]]
            senti['mentn_span'][0] = idx_mapping[senti['mentn_span'][0]]
            senti['mentn_span'][1] = idx_mapping[senti['mentn_span'][1]]

        res = []
        for i, senti in enumerate(senti_res):
            senti_st, senti_ed = senti['senti_span']
            senti_str = ''.join(words[senti_st:senti_ed+1])
            mentn_st, mentn_ed = senti['mentn_span']
            mentn_str = ''.join(words[mentn_st:mentn_ed+1])
            args = [{'name': senti['polarity'], 'str': mentn_str}]
            jo = {"senti_name": senti_str, "args": args, "senti_st": senti_st, "senti_ed": senti_ed}
            res.append(jo)

        return words, res


    # def put(self):
    #     print(request)
    #     segmented_text = request.form['segment'].split()
    #     if segmented_text == []:
    #         return {"srl": ""}
    #
    #     pred_idx_list = request.form['pred_list'].split()
    #     pred_idx_list = [int(id) for id in pred_idx_list]
    #     if pred_idx_list == []:
    #         return {"srl": ""}
    #
    #     return {"srl": str(srl)}

    def decode(self, segmented_test):
        pass

api.add_resource(CASAParser, '/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Should be consistent with training')
    parser.add_argument('--cuda_device', type=str, required=True, help='GPU ids (e.g. "1" or "1,2")')
    parser.add_argument('--bert_version', type=str, required=True, help='BERT version (e.g. "bert-base-chinese"')
    parser.add_argument('--tok2word_strategy', type=str, required=True, help='Should be consistent with training, e.g. avg')
    parser.add_argument('--mention_model_path', type=str, required=True, help='The saved mention model')
    parser.add_argument('--sentiment_model_path', type=str, required=True, help='The saved sentiment model')
    args, unparsed = parser.parse_known_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}'.format(device, n_gpu))

    tokenizer = BertTokenizer.from_pretrained(args.bert_version)

    print('Compiling model')
    mention_model = asa_model.BertAsaMe.from_pretrained(args.bert_version)
    mention_model.load_state_dict(torch.load(args.mention_model_path))
    mention_model.to(device)
    mention_model.eval()

    sentiment_model = asa_model.BertAsaSe.from_pretrained(args.bert_version)
    sentiment_model.load_state_dict(torch.load(args.sentiment_model_path))
    sentiment_model.to(device)
    sentiment_model.eval()
    print('Conversational Aspect Sentiment Analysis service is now available')
    app.run(host='0.0.0.0', port=8888)

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
