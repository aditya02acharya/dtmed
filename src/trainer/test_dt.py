import csv
import os
import torch

from src.data.patient_dataset import PatientDataset
from src.models.decision_transformer import DecisionTransformer
from src.utility.evaluation import evaluate_on_dataset


def test(args):
    # load data from this file
    dataset_path = f'{args.dataset_dir}/test.csv'
    log_csv_path = f'{args.log_dir}/evaluation.csv'

    task_name = args.task_name  # task name
    context_len = args.context_len  # K in decision transformer
    n_blocks = args.n_blocks  # num of transformer blocks
    embed_dim = args.embed_dim  # embedding (hidden) dim of transformer
    n_heads = args.n_heads  # num of transformer heads
    dropout_p = args.dropout_p  # dropout probability
    running_rtg = -1.0

    eval_chk_pt_path = os.path.join(args.chk_pt_dir, args.chk_pt_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    state_dim = 11
    act_dim = 238

    eval_model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=embed_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=dropout_p,
    ).to(device)

    eval_model.load_state_dict(torch.load(eval_chk_pt_path, map_location=device))

    print("model loaded from: " + eval_chk_pt_path)

    csv_writer = csv.writer(open(log_csv_path, 'a', 1))
    csv_header = (["pid", "timestep", "state_predicted", "state_truth", "action_predicted", "action_truth", "reward_predicted"])
    csv_writer.writerow(csv_header)

    traj_dataset = PatientDataset(dataset_path=dataset_path, 
                                  feature_path=os.path.join("experiments", task_name), 
                                  context_length=context_len)

    # evaluate on test dataset
    results = evaluate_on_dataset(traj_dataset, eval_model.eval(), running_rtg, device, context_len, state_dim,
                                  csv_writer, True)




