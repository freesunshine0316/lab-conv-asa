python asa_infer_e2e.py \
    --tok2word_strategy avg --cuda_device "7" --batch_size 10 \
    --bert_version bert-base-chinese \
    --mention_model_path logs/ASA.me_roberta_wwm.bert_model.bin \
    --sentiment_model_path logs/ASA.se_roberta_wwm.bert_model.bin \
    --in_path $1 \
    --out_path $2 --out_format full_text
