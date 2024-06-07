from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import utils
from model.helpers import cosine_beta_schedule, extract, apply_conditioning, Losses

Sample = namedtuple('Sample', 'trajectories chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, diffusion_dims, n_timesteps=1000,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True,
                 action_weight=1.0, loss_discount=1.0, loss_weights=None,
                 ):
        super().__init__()
        self.horizon = horizon
        self.diffusion_dims = diffusion_dims
        self.transition_dim = len(diffusion_dims)
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        # self.action_weight = action_weight

        # dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ### set loss coefficients for dimensions of observation
        # if weights_dict is None: weights_dict = {}
        # for ind, w in weights_dict.items():
        #    dim_weights[self.action_dim + ind] *= w

        ### decay loss with trajectory timestep: discount**t
        # discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        # discounts = discounts / discounts.mean()
        # loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ### manually set a0 weight
        # loss_weights[0, :self.action_dim] = action_weight
        # return loss_weights
        return torch.tensor(1.)

    # ------------------------------------------ sampling (old) ------------------------------------------#

    # def predict_start_from_noise(self, x_t, t, noise):
    #    '''
    #        if self.predict_epsilon, model output is (scaled) noise;
    #        otherwise, model predicts x0 directly
    #    '''
    #    if self.predict_epsilon:
    #        return (
    #                extract(self.sqrt_recip_alphas_cumprod, t, x_t[:, :, self.diffusion_dims].shape) * x_t[:, :, self.diffusion_dims] -
    #                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t[:, :, self.diffusion_dims].shape) * noise[:, :, self.diffusion_dims]
    #        )
    #    else:
    #        return noise
    #
    # def q_posterior(self, x_start, x_t, t):
    #    posterior_mean = x_start.clone()
    #    posterior_mean[:, self.diffusion_dims] = (
    #            extract(self.posterior_mean_coef1, t, x_start[:, :, self.diffusion_dims].shape) * x_start[:, :, self.diffusion_dims] +
    #            extract(self.posterior_mean_coef2, t, x_start[:, :, self.diffusion_dims].shape) * x_t[:, :, self.diffusion_dims]
    #    )
    #    posterior_variance = extract(self.posterior_variance, t, x_t.shape)
    #    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
    #    return posterior_mean, posterior_variance, posterior_log_variance_clipped
    #
    # def p_mean_variance(self, x, cond, t):
    #    x_recon = x.clone()
    #    x_recon[:, self.diffusion_dims] = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))
    #
    #    if self.clip_denoised:
    #        x_recon.clamp_(-1., 1.)
    #    else:
    #        assert RuntimeError()
    #
    #    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
    #        x_start=x_recon, x_t=x, t=t)
    #    return model_mean, posterior_variance, posterior_log_variance
    #
    # @torch.no_grad()
    # def p_sample_loop(self, init_trajectories, cond, verbose=True, return_chain=False, **sample_kwargs):
    #    device = self.betas.device
    #
    #    x = init_trajectories.clone()
    #    noise = torch.randn_like(x, device=device)
    #
    #    x[:, :, self.diffusion_dims] = noise[:, :, self.diffusion_dims]
    #    x = apply_conditioning(x, cond)
    #
    #    chain = [x] if return_chain else None
    #
    #    progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
    #    for i in reversed(range(0, self.n_timesteps)):
    #        t = make_timesteps(init_trajectories.shape[0], i, device)
    #        #todo: only diffusion dims?
    #        x = self.sample_fn(x, cond, t, **sample_kwargs)
    #        x = apply_conditioning(x, cond)
    #
    #        # progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
    #        if return_chain: chain.append(x)
    #
    #    progress.stamp()
    #
    #    #x, values = sort_by_values(x, values)
    #    if return_chain: chain = torch.stack(chain, dim=1)
    #    return Sample(x, chain)
    #
    # @torch.no_grad()
    # def conditional_sample(self, initial, cond, **sample_kwargs):
    #    '''
    #        conditions : [ (time, state), ... ]
    #    '''
    #    return self.p_sample_loop(initial, cond, **sample_kwargs)
    #
    # @torch.no_grad()
    # def sample_fn(self, x, cond, t):
    #    model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
    #    model_std = torch.exp(0.5 * model_log_variance)
    #
    #    noise = torch.zeros_like(x)
    #    # Get t for each index, set noise only if t!=0.
    #    noise[:, :, self.diffusion_dims] = torch.randn_like(x)[:, :, self.diffusion_dims]
    #    noise[t == 0] = 0
    #    return model_mean + model_std * noise

    # ------- sampling (again)  -----

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, init_trajectories, cond, verbose=True, return_chain=False, **sample_kwargs):
        device = self.betas.device

        batch_size = init_trajectories.shape[0]

        x = init_trajectories.clone()
        x[:, :, self.diffusion_dims] = torch.randn_like(x[:, :, self.diffusion_dims], device=device)
        if cond:
            x = apply_conditioning(x, cond)

        chain = [x.clone()] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            if cond:
                x = apply_conditioning(x, cond)
            x[:, :, self.diffusion_dims] = self.sample_fn(x, cond, t, **sample_kwargs)[:, :, self.diffusion_dims]
            progress.update({'t': i})
            if return_chain: chain.append(x.clone())

        progress.stamp()

        if return_chain: chain = torch.stack(chain, dim=1)
        return x, chain

    @torch.no_grad()
    def conditional_sample(self, initial, cond, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        return self.p_sample_loop(initial, cond, verbose=False, **sample_kwargs)

    @torch.no_grad()
    def sample_fn(self, x, cond, t):
        mean, _, log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        std = torch.exp(0.5 * log_variance)

        # no noise when t == 0
        noise = torch.randn_like(x)
        noise[t == 0] = 0

        return mean + std * noise

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = x_start.clone()
        sample[:, :, self.diffusion_dims] = (
                extract(self.sqrt_alphas_cumprod, t, x_start[:, :, self.diffusion_dims].shape) * x_start[:, :,
                                                                                                 self.diffusion_dims] +
                extract(self.sqrt_one_minus_alphas_cumprod, t, noise[:, :, self.diffusion_dims].shape) * noise[:, :,
                                                                                                         self.diffusion_dims]
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon[:, :, self.diffusion_dims], noise[:, :, self.diffusion_dims])
        else:
            loss, info = self.loss_fn(x_recon[:, :, self.diffusion_dims], x_start[:, :, self.diffusion_dims])

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, initial, cond, *args, **kwargs):
        return self.conditional_sample(initial, cond, *args, **kwargs)


class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)
