python server.py \
    --tok2word_strategy avg --cuda_device "7" --batch_size 10 \
    --bert_version bert-base-chinese \
    --mention_model_path logs/ASA.me_bert.bert_model.bin \
    --sentiment_model_path logs/ASA.se_bert.bert_model.bin
