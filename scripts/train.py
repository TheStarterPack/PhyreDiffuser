from data.sequence import SequenceDataset
from model.diffusion import GaussianDiffusion
from model.temporal import TemporalUnet
from utils import Trainer
from visualization.rendering import PhyreRenderer

observation_dim_per_object = 2
num_objects = 3

observation_dim = observation_dim_per_object * num_objects
horizon = 64

epochs = 100

dataset = SequenceDataset(
    "../../../Development/phyre-proj/PHYRE-diffusion/phyrediff/data/images/phyre_diff_00_1_task_100_actions_latents.h5")

model = TemporalUnet(horizon=horizon, transition_dim=observation_dim, cond_dim=observation_dim)

diffusion = GaussianDiffusion(model, horizon=horizon, observation_dim=observation_dim, action_dim=0)

trainer = Trainer(diffusion, dataset, PhyreRenderer(), sample_freq=1000)

for i in range(epochs):
    trainer.train(100)
