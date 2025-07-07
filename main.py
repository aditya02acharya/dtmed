import argparse

from src.test.test_dt import test
from src.train.train_dt import train


def main_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='task1')

    parser.add_argument('--max_eval_ep_len', type=int, default=5)
    parser.add_argument('--num_eval_ep', type=int, default=10)

    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='data/dt_runs')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)

    parser.add_argument('--max_train_iters', type=int, default=200)
    parser.add_argument('--num_updates_per_iter', type=int, default=100)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train(args)


def main_test():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', type=str, default='task1')
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--log_dir', type=str, default='data/dt_runs')

    parser.add_argument('--chk_pt_dir', type=str, default='data/saves/')
    parser.add_argument('--chk_pt_name', type=str,
                        default='dt_task1_30-03-22-14-56-07.pt')

    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--dropout_p', type=float, default=0.0)

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main_train()
