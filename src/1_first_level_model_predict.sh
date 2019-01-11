
#describe
#使用训练好的第一层模型进行预测，基于第一层模型的输出概率作为第二层模型的输入

#2 模型预测，共计25个基于bert的模型
export ZH_BERT_BASE_DIR=../data/bert/chinese_L-12_H-768_A-12 #google中文预训练bert地址
export EN_BERT_BASE_DIR=../data/bert/uncased_L-24_H-1024_A-16 #google英文文预训练bert地址
export EN_BERT_BASE_DIR2=../data/bert/uncased_L-12_H-768_A-12 #google英文文预训练bert地址
export MY_DATASET=../data/all #全局变量 数据集所在地址
###########################################################################################
#2-1 model (LNO 5)
echo '*******************start training model 01************************'
export OUTPUT_DIR=../data/model/output5
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-2 model (LNO.31)
echo '*******************start training model 02************************'
export OUTPUT_DIR=../data/model/output31
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-57079 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=false \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-3 model (LNO.47)
echo '*******************start training model 03************************'
export OUTPUT_DIR=../data/model/output47
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-31476 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv@enhanced_disagreed.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-4 model (LNO.48)
echo '*******************start training model 04************************'
export OUTPUT_DIR=../data/model/output48
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-31476 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=false \
  --train_file_list=train_dataset.csv@enhanced_disagreed.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-5 model (LNO.49)
echo '*******************start training model 05************************'
export OUTPUT_DIR=../data/model/output49
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-31476 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=mix \
  --train_file_list=train_dataset.csv@enhanced_disagreed.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-6 model (LNO.2)
echo '*******************start training model 06************************'
export OUTPUT_DIR=../data/model/output2
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=orignal \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-7 model (LNO.8)
echo '*******************start training model 07************************'
export OUTPUT_DIR=../data/model/output8
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-8 model (LNO.6)
echo '*******************start training model 08************************'
export OUTPUT_DIR=../data/model/output6
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-14274 \
  --max_seq_length=64 \
  --train_batch_size=64 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-9 model (LNO.9)
echo '*******************start training model 09************************'
export OUTPUT_DIR=../data/model/output9
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@True \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-10 model (LNO.17)
echo '*******************start training model 10************************'
export OUTPUT_DIR=../data/model/output17
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@True \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-11 model (LNO.23)
echo '*******************start training model 11************************'
export OUTPUT_DIR=../data/model/output23
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-57098 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@True \
  --filter_punct=true

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-12 model (LNO.10)
echo '*******************start training model 12************************'
export OUTPUT_DIR=../data/model/output10
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-55185 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@26.0@1.0@False \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-13 model (LNO.11)
echo '*******************start training model 13************************'
export OUTPUT_DIR=../data/model/output11
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-110371 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@26.0@1.0@True \
  --filter_punct=false

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-14 model (LNO.16)
echo '*******************start training model 14************************'
export OUTPUT_DIR=../data/model/output16
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-110371 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@26.0@1.0@True \
  --filter_punct=true

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-15 model (LNO.34)
echo '*******************start training model 15************************'
export OUTPUT_DIR=../data/model/output34
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-110315 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=false \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@26.0@1.0@True \
  --filter_punct=true

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-16 model (LNO.35)
echo '*******************start training model 16************************'
export OUTPUT_DIR=../data/model/output35
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-110315 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=mix \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@26.0@1.0@True \
  --filter_punct=true

cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-17 model (LNO.13)
echo '*******************start training model 17************************'
export OUTPUT_DIR=../data/model/output13
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-186733 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-18 model (LNO.18)
echo '*******************start training model 18************************'
export OUTPUT_DIR=../data/model/output18
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-186733 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-19 model (LNO.19)
echo '*******************start training model 19************************'
export OUTPUT_DIR=../data/model/output19
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-186733 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=true
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-20 model (LNO.20)
echo '*******************start training model 20************************'
export OUTPUT_DIR=../data/model/output20
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-186733 \
  --max_seq_length=96 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-21 model (LNO.14)
echo '*******************start training model 21************************'
export OUTPUT_DIR=../data/model/output14
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-186733 \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-22 model (LNO.21)
echo '*******************start training model 22************************'
export OUTPUT_DIR=../data/model/output21
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$ZH_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$ZH_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-110656 \
  --max_seq_length=48 \
  --train_batch_size=54 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=zh \
  --simple=true \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=2.0@78.0@1.0@True \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-23 model (LNO.en10)
echo '*******************start training model 23************************'
export OUTPUT_DIR=../data/model/output_en10
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$EN_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$EN_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-95163 \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=en \
  --simple=original \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@false \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-24 model (LNO.en07)
echo '*******************start training model 24************************'
export OUTPUT_DIR=../data/model/output_en07
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$EN_BERT_BASE_DIR2/vocab.txt \
  --bert_config_file=$EN_BERT_BASE_DIR2/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-28549 \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=en \
  --simple=original \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@false \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
###########################################################################################
#2-25 model (LNO.en11)
echo '*******************start training model 25************************'
export OUTPUT_DIR=../data/model/output_en11
python2 -u run_classifier.py \
  --task_name=p \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$EN_BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$EN_BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-57098 \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --gpu=0 \
  --language=en \
  --simple=original \
  --train_file_list=train_dataset.csv \
  --agree_disagree_unrelatead_sample_rate_abba=1.0@1.0@1.0@false \
  --filter_punct=false
cp $OUTPUT_DIR/eval_prob.tsv.* ../data/eval_prob/
cp $OUTPUT_DIR/test_results.tsv.* ../data/pred_prob/
