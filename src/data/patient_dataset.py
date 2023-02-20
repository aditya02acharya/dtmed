import os
import yaml
import errno
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.utility.data_utils import add_padding


class PatientDataset(Dataset):

    def __init__(self,
                 dataset_path: str = None,
                 feature_path: str = None,
                 context_length: int = 10,
                 scaling: bool = True):
        """
        Description:
            Torch Dataset class to load data from csv, apply min max scaling if required and
            deal with varying sequence lengths.

        Arguments:
            dataset_path: path to the data file on disk.
            feature_path: path to the data description file on disk.
            context_length: maximum sequence length.
            scaling: flag to enable/disable min-max scaling of covariates.
        """

        if os.path.exists(feature_path):
            with open(os.path.join(feature_path, "data_features.yml"), 'r') as file:
                self.data_config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(feature_path, "data_features.yml"))

        # parameters
        self.patient_identifier = self.data_config['pid']
        self.state_field = self.data_config['covariates']
        self.action_field = self.data_config['action']
        self.medication_field = self.data_config['medication']
        self.reward_field = self.data_config['rewards_to_go']
        self.reward_step_field = self.data_config['step_reward']
        self.action_dim = self.data_config['action_dim']
        self.scaler_range = self.data_config['scaler_config']
        self.context_len = context_length

        # load dataset from disk
        self.trajectories = pd.read_csv(dataset_path)

        # normalise state values
        if scaling:
            for col in self.state_field:
                self.trajectories[col] = (self.trajectories[col] - self.scaler_range[col][0]) / \
                                         (self.scaler_range[col][1] - self.scaler_range[col][0])

        # get unique patients id's in the data.
        self.trial_ids = np.unique(self.trajectories[self.patient_identifier])

    def __len__(self):
        return self.trial_ids.shape[0]

    def __getitem__(self, item):

        p_id = self.trial_ids[item]

        data = self.trajectories.loc[self.trajectories[self.patient_identifier].isin([p_id])]

        data_state = data[self.state_field].values
        data_action = data[self.action_field].values.reshape(-1)
        data_medication = data[self.medication_field].values.reshape(-1)
        data_reward = data[self.reward_field].values.reshape(-1)
        data_step_reward = data[self.reward_step_field].values.reshape(-1)
        traj_len = data.shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)
            states = torch.from_numpy(data_state[si: si + self.context_len])
            actions = torch.from_numpy(data_action[si: si + self.context_len])
            medications = torch.from_numpy(data_medication[si: si + self.context_len])
            returns_to_go = torch.from_numpy(data_reward[si: si + self.context_len])
            step_reward = torch.from_numpy(data_step_reward[si: si + self.context_len])
            timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            # add zero padding to ensure equal sequence lengths.
            padding_len = self.context_len - traj_len

            states = torch.from_numpy(np.array(data_state))
            states = add_padding(states, padding_len)

            actions = torch.from_numpy(data_action)
            actions = add_padding(actions, padding_len)

            medications = torch.from_numpy(data_medication)
            medications = add_padding(medications, padding_len)

            returns_to_go = torch.from_numpy(data_reward)
            returns_to_go = add_padding(returns_to_go, padding_len)

            step_reward = torch.from_numpy(data_step_reward)
            step_reward = add_padding(step_reward, padding_len)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)], dim=0)

        return timesteps, states, actions, medications, returns_to_go, traj_mask, p_id, step_reward
