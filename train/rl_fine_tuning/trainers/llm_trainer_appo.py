import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mat.utils.util import get_gard_norm, huber_loss, mse_loss


class APPOTrainer:

    def __init__(self, args, agent, num_agents):
        self.tpdv = dict(dtype=torch.float32, device=torch.device("cuda:0"))
        self.agent = agent

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.entropy_coef = args.entropy_coef
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        self.policy_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()),
            lr=self.lr,
            eps=1e-5,
            weight_decay=0,
        )
        self.critic_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()),
            lr=self.critic_lr,
            eps=1e-5,
        )

    def cal_policy_loss(
        self, log_prob_infer, log_prob_batch, advantages_batch, entropy
    ):

        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)

        approx_kl = ((imp_weights - 1) - log_ratio).mean()

        surr1 = (
            -torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages_batch
        )
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)
        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl

    def cal_value_loss(self, values_infer, value_preds_batch, return_batch):

        value_pred_clipped = value_preds_batch + (
            values_infer - value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)
        error_clipped = return_batch - value_pred_clipped
        error_unclipped = return_batch - values_infer
        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_unclipped = huber_loss(error_unclipped, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_unclipped = mse_loss(error_unclipped)
        value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
        return value_loss * self.value_loss_coef

    def ppo_update(self, sample):
        (
            obs_batch,
            action_batch,
            log_prob_batch,
            value_preds_batch,
            return_batch,
            advantages_batch,
            action_tokens_batch,
        ) = sample

        log_prob_batch = torch.from_numpy(log_prob_batch).to("cuda")
        value_preds_batch = torch.from_numpy(value_preds_batch).to("cuda")
        return_batch = torch.from_numpy(return_batch).to("cuda")
        advantages_batch = torch.from_numpy(advantages_batch).to("cuda")
        action_tokens_batch = torch.from_numpy(action_tokens_batch).to("cuda")
        batch_size = obs_batch.shape[0]

        # critic update
        values_infer = self.agent.get_action_values(np.concatenate(obs_batch))
        values_infer = values_infer.view(batch_size, -1)

        value_loss = self.cal_value_loss(values_infer, value_preds_batch, return_batch)
        # print("value_loss: ", value_loss)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.agent.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.agent.critic.parameters())
        self.critic_optimizer.step()
        value_loss = value_loss.item()
        self.critic_optimizer.zero_grad()
        critic_grad_norm = critic_grad_norm.item()

        # policy update
        self.policy_optimizer.zero_grad()
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        total_approx_kl = 0
        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            log_prob_infer, entropy = self.agent.infer_for_action_update(
                np.concatenate(obs_batch[start:end]),
                action_tokens_batch[start:end].view(-1, action_tokens_batch.shape[-1]),
            )

            log_prob_infer = log_prob_infer.view(obs_batch[start:end].shape[0], -1)

            cp_adv_batch = advantages_batch[start:end]
            cp_adv_batch = (cp_adv_batch - cp_adv_batch.mean()) / (
                cp_adv_batch.std() + 1e-8
            )

            entropy = entropy.view(obs_batch[start:end].shape[0], -1)
            policy_loss, approx_kl = self.cal_policy_loss(
                log_prob_infer, log_prob_batch[start:end], cp_adv_batch, entropy
            )
            total_approx_kl += approx_kl / self.gradient_cp_steps

            # print("policy_loss: ", policy_loss)

            policy_loss /= self.gradient_cp_steps
            policy_loss.backward()
        if total_approx_kl > 0.02:
            self.policy_optimizer.zero_grad()
            return value_loss, critic_grad_norm, 0, 0

        policy_grad_norm = nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), self.max_grad_norm
        )
        self.policy_optimizer.step()
        policy_loss = policy_loss.item()
        self.policy_optimizer.zero_grad()
        policy_grad_norm = policy_grad_norm.item()

        return value_loss, critic_grad_norm, policy_loss, policy_grad_norm

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["value_loss"] = 0
        train_info["value_grad_norm"] = 0
        train_info["policy_loss"] = 0
        train_info["policy_grad_norm"] = 0

        update_time = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.appo_sampler(self.num_mini_batch)
            for sample in data_generator:
                value_loss, value_grad_norm, policy_loss, policy_grad_norm = (
                    self.ppo_update(sample)
                )
                train_info["value_loss"] += value_loss
                train_info["value_grad_norm"] += value_grad_norm
                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                update_time += 1

        for k in train_info.keys():
            train_info[k] /= update_time

        return train_info

    def prep_training(self):
        self.agent.actor().train()
        self.agent.critic().train()

    def prep_rollout(self):
        self.agent.actor().eval()
        self.agent.critic().eval()
