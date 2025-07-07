import os
import csv
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.patient_dataset import PatientDataset
from src.models.decision_transformer import DecisionTransformer

torch.set_printoptions(threshold=10_000)


def train(args):
    # training configs
    task_name = args.task_name  # task name

    batch_size = args.batch_size  # training batch size
    lr = args.lr  # learning rate
    wt_decay = args.wt_decay  # weight decay
    warmup_steps = args.warmup_steps  # warmup steps for lr scheduler

    # total updates = max_train_iters x num_updates_per_iter
    max_train_iters = args.max_train_iters
    num_updates_per_iter = args.num_updates_per_iter

    context_len = args.context_len  # K in decision transformer
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    n_heads = args.n_heads  # num of transformer heads
    dropout_p = args.dropout_p  # dropout probability

    state_dim = 11
    act_dim = 273


    # load data from this file
    dataset_path = f'{args.dataset_dir}/train.csv'

    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # training device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%d-%m-%y-%H-%M-%S")

    prefix = "dt_" + task_name

    save_model_name = prefix + "_model_" + start_time_str + ".pt"
    save_model_path = os.path.join(args.dataset_dir, "saves", save_model_name)
    save_best_model_path = save_model_path[:-3] + "_best.pt"

    log_csv_name = prefix + "_log_" + start_time_str + ".csv"
    log_csv_path = os.path.join(log_dir, log_csv_name)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["duration", "num_updates", "action_loss"])

    csv_writer.writerow(csv_header)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    print("device set to: " + str(device))
    print("dataset path: " + dataset_path)
    print("model save path: " + save_model_path)
    print("log csv save path: " + log_csv_path)

    traj_dataset = PatientDataset(dataset_path=dataset_path, 
                                  feature_path=os.path.join("experiments", task_name), 
                                  context_length=context_len)
    traj_data_loader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    data_iter = iter(traj_data_loader)

    model = DecisionTransformer(state_dim=state_dim, act_dim=act_dim, n_blocks=n_blocks, h_dim=embed_dim,
                                context_len=context_len, n_heads=n_heads, drop_p=dropout_p).to(device)
    model = model.float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    total_updates = 0

    for i_train_iter in range(max_train_iters):

        log_action_losses = []
        model.train()

        for _ in range(num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask, _, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask, _, _ = next(data_iter)

            timesteps = timesteps.to(device)  # B x T
            states = states.to(device)  # B x T x state_dim
            actions = actions.to(device)  # B x T x act_dim
            returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1)  # B x T x 1
            traj_mask = traj_mask.to(device)  # B x T
            action_target = torch.clone(actions).detach().to(device)

            try:
                state_preds, action_preds, return_preds = model.forward(
                    timesteps=timesteps.int(),
                    states=states.float(),
                    actions=actions.int(),
                    returns_to_go=returns_to_go.float()
                )
                # only consider non padded elements
                action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
                action_target = action_target.view(-1, 1)[traj_mask.view(-1, ) > 0]
                # print(action_preds, action_target.reshape(-1))
                action_loss = F.cross_entropy(action_preds, action_target.reshape(-1), reduction='mean')
                # action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())

            except Exception:
                print(actions.int())

        mean_action_loss = np.mean(log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - start_time)

        total_updates += num_updates_per_iter

        log_str = ("=" * 60 + '\n' +
                   "time elapsed: " + time_elapsed + '\n' +
                   "num of updates: " + str(total_updates) + '\n' +
                   "action loss: " + format(mean_action_loss, ".5f") + '\n'
                   )

        print(log_str)

        log_data = [time_elapsed, total_updates, mean_action_loss]

        csv_writer.writerow(log_data)

        # save model
        print("saving current model at: " + save_model_path)
        torch.save(model.state_dict(), save_model_path)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("saved last updated model at: " + save_model_path)
    print("=" * 60)
