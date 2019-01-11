#模型第二层和第三层
echo 'starting blending and stacking'
python2 blending_and_stacking.py ../data/eval_prob ../data/pred_prob 25 true
echo 'done'