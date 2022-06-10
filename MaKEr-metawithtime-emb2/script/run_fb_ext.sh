#!/bin/sh

kge_model=['TComplEx', 'DistMult', 'ComplEx', 'RotatE','TTransE','T-DistMult','TComplEx','TeRo']
gamma_all = [0.001,0.01,0.1,1,10,100]
reg_all = [0,0.001,0.0025,0.005,0.0075,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
lr_all = [0.0001,0.001,0.01,0.1]
gpu=0
for kge in kge_model:
  for gamma in gamma_all:
      for reg in reg_all:
          for lr in lr_all:
              python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma ${gamma} --reg ${reg} --lr ${lr}
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 1 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 10 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 100 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 0.1 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 10 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 10 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 10 --reg 1 --lr 0.01
#python main.py --data_path ./test_data.pkl --task_name icews_transe --kge ${kge} --gpu cuda:${gpu} --gamma 10 --reg 1 --lr 0.01