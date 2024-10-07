import time
import os
import numpy as np
from functools import reduce
import torch
from tensorboardX import SummaryWriter
from mat.agents.qwen_lora_agent import QwenLoRAgent
from mat.models.rm import ProcessRM
from mat.utils.language_buffer import LanguageBuffer
from mat.trainers.llm_trainer_appo import APPOTrainer
from mat.trainers.llm_trainer_tppo import TPPOTrainer

def _t2n(x):
    return x.detach().cpu().numpy()

class MathRunner:
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.save_interval = self.all_args.save_interval
        self.algo = self.all_args.algorithm_name

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.agent = QwenLoRAgent(self.all_args.model_name_or_path, self.all_args.max_new_tokens, self.algo)
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        self.prm = ProcessRM(self.all_args.prm_model_name_or_path)

        if self.algo == "APPO":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo == "TPPO":
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError

    def run(self):
        obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        episodic_returns = []
        for episode in range(episodes):
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads  
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)
                
                # output rewards
                rewards = self.prm.get_reward(obs, actions)

                # Obs reward and next obs
                obs, fake_rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = obs, rewards, dones, values, actions, action_tokens, log_probs
                self.insert(data)
                
                for i in range(self.n_rollout_threads):
                    if dones[i, 0]:
                        episodic_returns.append(rewards[i, 0])

            # compute return and update network
            self.before_update()
            train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()
            
            # save model
            if (episode == episodes - 1 or episode % self.save_interval == 0):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                print("average_step_rewards: ", np.mean(self.buffer.rewards))
                train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                train_infos["average_currect_rate"] = np.mean(episodic_returns)
                self.log_infos(train_infos, total_num_steps)
                episodic_returns = []

            # eval
            # if self.all_args.use_eval and episode % self.eval_interval == 0:
            #     self.eval(total_num_steps)
        

    @torch.no_grad()
    def collect(self, step):
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[step]))
        
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def insert(self,data):
        obs, rewards, dones, values, actions, action_tokens, log_probs = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)

        if self.algo == "APPO":
            self.buffer.insert_appo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo == "TPPO":
            self.buffer.insert_tppo(obs, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[-1]))
        next_values = np.array(np.split(next_values, self.n_rollout_threads))
        if self.algo == "APPO":
            self.buffer.batch_process_appo(next_values)
        elif self.algo == "TPPO":
            self.buffer.batch_process_tppo(next_values)
        else:
            raise NotImplementedError

    def log_infos(self, infos, total_num_steps):
        for k, v in infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episodic_returns = []

        eval_obs = self.eval_envs.reset()
        while True:
            eval_actions, _ = self.agent.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            eval_obs, eval_rewards, eval_dones, _ = self.eval_envs.step(eval_actions)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i, 0]:
                    eval_episode += 1
                    eval_episodic_returns.append(eval_rewards[eval_i])

            if eval_episode >= self.all_args.eval_episodes:
                eval_currect_rate = np.mean(eval_episodic_returns)
                env_infos = {'eval_currect_rate': eval_currect_rate}     
                print("total_num_steps: ", total_num_steps)
                print("eval_currect_rate is {}.".format(eval_currect_rate))           
                self.log_infos(env_infos, total_num_steps)
                break
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.agent.restore(model_dir)


