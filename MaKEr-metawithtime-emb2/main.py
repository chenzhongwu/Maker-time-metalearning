import argparse
from utils import init_dir
from meta_trainer import MetaTrainer
import os
from subgraph import gen_subgraph_datasets
#   用icews，RPG不带时间，GNN带时间
#   另一处的对比实验是用原框架，不带时间
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./test_data.pkl')
    parser.add_argument('--state_dir', default='./state')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--tb_log_dir', default='./tb_log')

    parser.add_argument('--task_name', default='icews14')
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--num_exp', default=1, type=int)

    parser.add_argument('--train_bs', default=64, type=int)
    parser.add_argument('--eval_bs', default=16, type=int)
    # parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--log_per_step', default=10, type=int)
    parser.add_argument('--check_per_step', default=30, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--num_sample_cand', default=5, type=int)

    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--ent_dim', default=None, type=int)
    parser.add_argument('--rel_dim', default=None, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rel_bases', default=4, type=int)
    parser.add_argument('--num_time_bases', default=3, type=int)

    # parser.add_argument('--kge', default='TeRo', type=str,
    #                     choices=['TComplEx', 'DistMult', 'ComplEx', 'RotatE','TTransE','T-DistMult','TComplEx','TeRo'])
    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--adv_temp', default=1, type=float)
    # parser.add_argument('--gamma', default=10, type=float)
    parser.add_argument('--xishu', default=0.5, type=float)
    # parser.add_argument('--reg', default=1, type=float)

    parser.add_argument('--cpu_num', default=10, type=float)
    parser.add_argument('--gpu', default='cuda:1', type=str)

    # subgraph
    parser.add_argument('--db_path', default=None)
    parser.add_argument('--num_train_subgraph', default=10000)
    parser.add_argument('--num_sample_for_estimate_size', default=10)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)

    # time
    parser.add_argument('--num_time', default=365, type=int)
    # parser.add_argument('--time_dim', default=128, type=int)
    # parser.add_argument('--', default=, type=int)

    args = parser.parse_args()


    args.db_path = args.data_path + '_subgraph'

    print(args.db_path)
    # if  os.path.exists(args.db_path):
    #     print("156483")
    if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args)

    init_dir(args)
    # print("156483")
    kge_model = [  'DistMult', 'T-DistMult','TeRo','TComplEx', 'ComplEx', 'RotatE', 'TTransE', 'TComplEx']
    gamma_all = [0.001, 0.01, 0.1, 1 ]
    reg_all = [0, 0.001,  0.005,  0.01,  0.05, 0.1]
    lr_all = [1e-4, 1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 0.00000001, 0.0000001, 0.000001, 0.00001]
    gpu = 0
    for args.kge in kge_model:
        for args.gamma in gamma_all:
            for args.reg in reg_all:
                for args.lr in lr_all:
                    if args.kge in ['TransE', 'DistMult', 'TTransE', 'T-DistMult']:
                        args.ent_dim = args.dim
                        args.rel_dim = args.dim
                        args.time_dim = args.dim
                    elif args.kge == 'RotatE':
                        args.ent_dim = args.dim * 2
                        args.rel_dim = args.dim * 2
                    elif args.kge == 'TeRo':
                        args.ent_dim = args.dim * 2
                        args.rel_dim = args.dim
                        args.time_dim = args.dim* 2
                    elif args.kge == 'ComplEx':
                        args.ent_dim = args.dim * 2
                        args.rel_dim = args.dim * 2
                    elif args.kge == 'TComplEx':
                        args.ent_dim = args.dim * 2
                        args.rel_dim = args.dim * 2
                        args.time_dim = args.dim * 2
                    for run in range(args.num_exp):
                        args.run = run
                        # print("===========")
                        # print(args.run)
                        args.exp_name = args.task_name + "_run" + str(args.run)

                        trainer = MetaTrainer(args)
                        trainer.train()

                        del trainer
