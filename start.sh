#python main.py -gpu 2 -props ./properties/prob.sample > result
#python pretrain.py -gpu 1 -props ./properties/prob.pretrain > result
#python reinforcement_learning.py -gpu 2 -props ./properties/prob.rl > result.rl
python rl.py -gpu 1 -props ./properties/prob.rl > result.rl
