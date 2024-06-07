import torch
import sys
sys.path.append('../')

from data.sequence import SequenceDataset
from model.diffusion import GaussianDiffusion
from model.temporal import TemporalUnet
from utils import Trainer
from visualization.rendering import SimplePhyreRenderer, PhyreTrajectoryRenderer
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

observation_dim_per_object = 15
num_objects = 3

train_dataset = "../datasets/phyre_diff_00_1_tasks_1_action_latents.h5"

observation_dim = observation_dim_per_object * num_objects
horizon = 64

epochs = 1000

dataset = SequenceDataset(train_dataset)

model = TemporalUnet(horizon=horizon, transition_dim=observation_dim, cond_dim=observation_dim)

diffusion = GaussianDiffusion(model, horizon=horizon, diffusion_dims=(0, 1, 15, 16, 30, 31), n_timesteps=28).to(device)

trainer = Trainer(diffusion, dataset, PhyreTrajectoryRenderer(), sample_freq=10000, log_freq=1000,
                  save_freq=10000, device=device)

for i in tqdm(range(epochs)):
    trainer.train(100)
