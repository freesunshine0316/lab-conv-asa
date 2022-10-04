python asa_infer_e2e.py \
    --tok2word_strategy avg --cuda_device $1 --batch_size 10 \
    --bert_version hfl/chinese-bert-wwm-ext \
    --mention_model_path logs/ASA.me_bert_wwm_tok_v2.checkpoint.bin \
    --sentiment_model_path logs/ASA.se_bert_wwm_tok_v2.checkpoint.bin \
    --in_path $2 \
    --out_path $3 --out_format full_text
