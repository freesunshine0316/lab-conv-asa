import json


def process_naturalconv(inpath, outpath):
    speaker_map = {0: 'A:', 1: 'B:'}
    with open(inpath, "r") as infile, open(outpath, "w") as outfile:
        data = json.load(infile)
        for line in data:
            line_str = '\n'.join(f'{speaker_map[i%2]} {turn}' for i, turn in enumerate(line["content"]))
            outfile.write(line_str + '\n\n')


def process_norm(inpath, outpath, buff_size=1000000):
    speaker_map = {0: 'A:', 1: 'B:'}
    with open(inpath, "r") as infile, open(outpath, "w") as outfile:
        data = []
        conv = []
        for line in infile:
            line = json.loads(line.strip())
            if line["turn_id"] == 0:  # first turn of a conv
                if len(conv) > 0:
                    data.append(conv)
                    conv = []
            if len(data) > buff_size:
                for inst in data:
                    inst_str = '\n'.join(f'{speaker_map[i%2]} {sent}' for i, sent in enumerate(inst))
                    outfile.write(inst_str + '\n\n')
                data = []
            conv.append(line["query"].replace("[", "{").replace("]", "}"))
        if len(data) > 0:
            for inst in data:
                inst_str = '\n'.join(f'{speaker_map[i%2]} {sent}' for i, sent in enumerate(inst))
                outfile.write(inst_str + '\n\n')
            data = []


if __name__ == '__main__':
    # process_norm("data/chat_270m_utf8.jsonl", "data/chat_270m_utf8.txt")
    # process_norm("data/dulemon_both_dev.jsonl", "data/dulemon_both_dev.txt")
    # process_norm("data/dulemon_both_train.jsonl", "data/dulemon_both_train.txt")
    process_naturalconv("data/naturalconv_dialog.json", "data/naturalconv_dialog.txt")
    pass
