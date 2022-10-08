
import os, sys, json

def remove_space(sent):
    is_last_ascii = False
    new_sent = []
    for tok in sent.split():
        if len(new_sent) == 0:
            new_sent.append(tok)
        elif is_last_ascii and tok.isascii():
            new_sent.append(tok)
        else:
            new_sent[-1] = new_sent[-1] + tok
        is_last_ascii = tok.isascii()
    return ' '.join(new_sent)

def process_file(inpath, outpath, start_id):
    with open(inpath, 'r') as infile, open(outpath, 'w') as outfile:
        for i, line in enumerate(infile):
            line = json.loads(line.strip())
            txt = '\n'.join([remove_space(x) for x in line["conv"]])
            obj = {'id': start_id + i, 'txt': txt}
            outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

process_file('trial_120.jsonl', 'trial_120.jsonl_proc', 0)
