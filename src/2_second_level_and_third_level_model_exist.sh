#模型第二层和第三层
echo 'starting blending and stacking'

# 直接用我们训练好的第一层模型结果进行融合
python2 blending_and_stacking.py ../data/eval_prob_exist ../data/pred_prob_exist 25 False

echo 'done'