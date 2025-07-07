import tqdm
import torch
from torch.utils.data import DataLoader


def evaluate_on_dataset(dataset, model, running_rtg, device, context_len, state_dim, csv_writer=None, test=False):
    num_iteration = len(dataset)

    traj_data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False)
    data_iter = iter(traj_data_loader)

    results = {}

    total_reward = 0

    total_timesteps = 0

    timesteps_db = torch.arange(start=0, end=context_len, step=1)
    timesteps_db = timesteps_db.repeat(1, 1).to(device)

    with torch.no_grad():

        for _ in tqdm.tqdm(range(num_iteration)):
            timesteps, states, actions, returns_to_go, traj_mask, p_id, step_rwd = next(data_iter)
            max_ep_len = len((traj_mask == 1).nonzero(as_tuple=True)[0])

            # zeros place holders
            actions_db = torch.zeros((1, 40, 1),
                                     dtype=torch.float32, device=device)
            states_db = torch.zeros((1, 40, state_dim),
                                    dtype=torch.float32, device=device)
            rewards_to_go_db = torch.zeros((1, 40, 1),
                                           dtype=torch.float32, device=device)

            # init episode
            running_reward = 0
            running_state = None
            actions_db[0, 0] = actions[0, 0].to(device)
            states_db[0, 0] = states[0, 0].to(device)
            timesteps = timesteps.to(device)
            state_preds = None
            action_preds = None

            for t in range(max_ep_len):
                total_timesteps += 1

                # calculate running rtg
                running_rtg = running_rtg - running_reward
                rewards_to_go_db[0, t] = running_rtg

                if t < context_len:
                    state_preds, action_preds, return_preds = model.forward(timesteps_db[:, :context_len].int(),
                                                                            states_db[:, :context_len],
                                                                            actions_db[:, :context_len].int(),
                                                                            rewards_to_go_db[:, :context_len])

                    act = torch.argmax(action_preds[0, t].detach())
                    running_state = state_preds[0, t].detach()
                    running_reward = step_rwd[0, t].to(device)  # return_preds[0, t].detach()
                else:
                    break

                if test:
                    log_data = [p_id, t, str(running_state.tolist()), str(states[0, t].tolist()), str(act.item()),
                                str(actions[0, t].tolist()), str(running_reward.item())]
                    csv_writer.writerow(log_data)

                states_db[0, t] = states[0, t].to(device)
                actions_db[0, t] = actions[0, t].to(device)

                total_reward += running_reward

            results['eval/avg_reward'] = total_reward / num_iteration
            results['eval/avg_ep_len'] = total_timesteps / num_iteration

    return results
