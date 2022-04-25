python server_v2.py \
    --tok2word_strategy avg --cuda_device "4" --batch_size 10 \
    --bert_version hfl/chinese-bert-wwm-ext \
    --mention_model_path logs/ASA.me_bert_wwm_tok.checkpoint.bin \
    --sentiment_model_path logs/ASA.se_bert_wwm_tok.checkpoint.bin
