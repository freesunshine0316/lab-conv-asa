python asa_infer_e2e.py \
    --tok2word_strategy avg --cuda_device "0" --batch_size 10 \
    --bert_version hfl/chinese-bert-wwm-ext \
    --mention_model_path logs/ASA.me_bert_wwm_tok.checkpoint.bin \
    --sentiment_model_path logs/ASA.se_bert_wwm_tok.checkpoint.bin \
    --in_path $1 \
    --out_path $2 --out_format full_text
