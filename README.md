# ssm-planning-exp

Learn the dynamics through a latent state-space model and plan the ego vehicle's action through CEM. The observation is a feature vector extracted from a pre-trained attention module.

## Pre-trained attention module

The fixed `attn_feature.pt` that is used in the transform method of `dataset.py` is pre-trained from a behavioral cloning task.

## Loss function

My first attempt is to sample the log_probs, what I plan to do is to figure out an analytical expression to avoid sampling (hence enhance efficiency and robustness).

## Result

Pending

