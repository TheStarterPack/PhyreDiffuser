from data.sequence import SequenceDataset
from model.diffusion import GaussianDiffusion
from model.temporal import TemporalUnet
from utils import Trainer
from visualization.rendering import SimplePhyreRenderer, PhyreTrajectoryRenderer

observation_dim_per_object = 15
num_objects = 3

observation_dim = observation_dim_per_object * num_objects
horizon = 64

epochs = 100

dataset = SequenceDataset("../datasets/phyre_diff_00_10_tasks_2_actions_latents.h5")

model = TemporalUnet(horizon=horizon, transition_dim=observation_dim, cond_dim=observation_dim)

diffusion = GaussianDiffusion(model, horizon=horizon, diffusion_dims=(0, 1, 15, 16, 30, 31))

trainer = Trainer(diffusion, dataset, PhyreTrajectoryRenderer(), sample_freq=1000)

for i in range(epochs):
    trainer.train(100)
