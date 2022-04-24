python server_v2.py \
    --tok2word_strategy avg --cuda_device "4" --batch_size 10 \
    --bert_version bert-base-chinese \
    --mention_model_path logs/ASA.me_roberta_wwm.bert_model.bin \
    --sentiment_model_path logs/ASA.se_roberta_wwm.bert_model.bin
