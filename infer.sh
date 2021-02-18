python asa_infer_e2e.py \
    --tok2word_strategy avg --cuda_device "2" --batch_size 10 \
    --bert_version bert-base-chinese \
    --mention_model_path logs/ASA.me_bert.bert_model.bin \
    --sentiment_model_path logs/ASA.se_bert.bert_model.bin \
    --in_path data/demo.txt \
    --out_path data/demo.rst --out_format casa_json
