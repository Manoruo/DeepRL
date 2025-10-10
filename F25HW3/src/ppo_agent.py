# ppo_agent.py
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from buffer import Buffer
from policies import ActorCritic

class PPOAgent:
    def __init__(self, env_info, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 clip_coef=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 update_epochs=10, minibatch_size=64, rollout_steps=4096, device="cpu"):
        self.device = torch.device(device)
        policy = ActorCritic(
            env_info["obs_dim"],
            env_info["act_dim"],
            env_info["act_low"],
            env_info["act_high"],
            hidden=(64, 64),
        )
        self.actor = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        
        # PPO with KL penalty parameters
        self.beta = .5  # Initial KL penalty coefficient
        self.target_kl = 0.01  # Target KL divergence
        
        # Internal state for rollout collection
        self._curr_policy_rollout = []
        self._rollout_buffer = Buffer(
            size=rollout_steps*50,
            obs_dim=policy.obs_dim,
            act_dim=policy.act_dim,
            device=device
        )
        self._steps_collected_with_curr_policy = 0
        self._policy_iteration = 1
    
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist, value = self.actor(obs_t) # Query actor-crtic for action dist (a_t) and value (v_st)
            action = dist.sample() # sample action from distribtuion      
            log_prob = dist.log_prob(action) # save probability of action under the current policy (this function just maps the action to the log prob under the current distribution)
            
            return {
                "action": action.squeeze(0).cpu().numpy(),
                "log_prob": float(log_prob.squeeze(0).item()),
                "value": float(value.squeeze(0).item())
            }

    def step(self, transition: Dict[str, Any]) -> Dict[str, float]:
        """
        PPO-specific step: collect transitions until rollout is full, then update.
        
        transition should contain:
        - obs, action, reward, next_obs, done, truncated
        - log_prob, value (from act() call)
        """
        # Add to current rollout
        self._curr_policy_rollout.append(transition.copy())
        self._steps_collected_with_curr_policy += 1
        stop = transition['done'] or transition['truncated']
        ret = {}
        # ---------------- Problem 1.3.1: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.1 ###
        if stop: 
           
            # Compute GAE and returns of current trajectory/rollout
            advantages, returns = self._compute_gae(self._curr_policy_rollout)

            # Prepare batch and store in rollout buffer
            batch = self._prepare_batch(advantages, returns)

            # Add to rollout buffer
            self._rollout_buffer.add_batch(batch)

            # Check if temp rollout buffer is full enough for update. I.e check if we have collected enough transitions with current policy before updating
            if self._steps_collected_with_curr_policy >= self.rollout_steps:
                # Perform PPO update
                stats = self._perform_update()
                ret.update(stats)

                # Reset current rollout
                self._curr_policy_rollout = []
                self._steps_collected_with_curr_policy = 0
                self._policy_iteration += 1
        
        ### END STUDENT SOLUTION - 1.3.1 ###

        return ret  # Leave this as an empty dictionary if no update is performed

    def _perform_update(self) -> Dict[str, float]:
        """Perform PPO update using collected rollout"""
        all_stats = []

        # To log metrics correctly, make sure you have the following lines in this function
        # loss, stats = self._ppo_loss(minibatch)
        # all_stats.append(stats)
        
        ### EXPERIMENT 1.6 CODE ###
        
        on_policy_batch = self._rollout_buffer.sample(self.rollout_steps, filter={"iteration": [self._policy_iteration]})
        off_policy_batch = self._rollout_buffer.sample(self.rollout_steps)
        half_off_policy_batch = {k: v[:len(v)//2] for k, v in off_policy_batch.items()}
        half_on_policy_batch = {k: v[:len(v)//2] for k, v in on_policy_batch.items()}
        
        # combine half on-policy and half off-policy


        #full_batch = off_policy_batch
        ### EXPERIMENT 1.6 CODE END ###
        
        # ---------------- Problem 1.3.2: PPO Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.3.2 ###
        
        # get full batch to break up latter
        full_batch = on_policy_batch
        
        # normalize advantages
        advantages = full_batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_batches = np.ceil(self.rollout_steps / self.minibatch_size)

        for epoch in range(self.update_epochs):
            # smarter way to sample minibatches 
            total_batches = np.ceil(self.rollout_steps / self.minibatch_size)
            perm = torch.randperm(len(full_batch["advantages"]), device=self.device)
            for i in range(int(total_batches)):
                
                idxs = perm[i * self.minibatch_size : (i + 1) * self.minibatch_size]

                minibatch = {k: v[idxs] for k, v in full_batch.items()}

                # Compute loss and perform gradient step
                loss, stats = self._ppo_loss(minibatch)
                all_stats.append(stats)
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for stability (even though surrogate clips, advantage can still be huge)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()


        ### END STUDENT SOLUTION - 1.3.2 ###
        
        # ---------------- Problem 1.4.2: KL Divergence Beta Update ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###
        kl_aprox = np.mean([s["kl"] for s in all_stats])
        if kl_aprox < self.target_kl / 1.5:
            self.beta = self.beta / 2 
        elif kl_aprox > self.target_kl * 1.5:
            self.beta = self.beta * 2
        ### END STUDENT SOLUTION - 1.4.2 ###
        
        if all_stats:
            return {k: np.mean([s[k] for s in all_stats]) for k in all_stats[0].keys()}
        else:
            return {}
        
    def _compute_gae(self, rollout) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rollout)
        rewards = np.array([t["reward"] for t in rollout])
        values = np.array([t["value"] for t in rollout])
        dones = np.array([t["done"] for t in rollout])  # Get done flag for each timestep
        
        # Get the final value for bootstrap (v_s_T)
        next_obs = rollout[-1]["next_obs"]
        with torch.no_grad():
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, final_v = self.actor(obs_t)
            final_v = float(final_v.squeeze(0).item())
        
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # ---------------- Problem 1.2: Compute GAE ----------------
        ### BEGIN STUDENT SOLUTION - 1.2 ###
        A_GAE = 0
        for t in reversed(range(T)): 
            if t == T - 1:
                # if last step, use final_v (if it was to continue what would the value be? if end then its 0)
                next_value = final_v
            else:
                next_value = values[t + 1] # compute v_t+1
            
            next_non_terminal = 1.0 - dones[t]

            # Here we estimate Gt ~ r_t + gamma * v_t+1 (typically estimate v_t+1 with a neural net)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t] # Here we comptute G_t using TD(1) to determine advantage
            
            # Delta is our one step estimate of advantage (i.e grounded with 1 step reward) but we need to apply a weight to it to "smooth" it out
            A_GAE = delta + self.gamma * self.gae_lambda * next_non_terminal * A_GAE # note we can unroll this recursively so we get a better estimate of advantage as we go further back in time
            # A_GAE makese sense because its just "expected" advantage and it gets more accurate as we go further back since we unroll more rewards

            advantages[t] = A_GAE
            returns[t] = advantages[t] + values[t]
        
        ### END STUDENT SOLUTION - 1.2 ###
        
        return advantages, returns
    
    def _ppo_loss(self, batch):
        """Standard PPO loss computation"""
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        
        # Forward pass
        dist, values = self.actor(obs)
        log_probs = dist.log_prob(actions)
        if log_probs.ndim > 1:
            log_probs = log_probs.sum(dim=-1)
            
        policy_loss = 0 # Placeholder
        total_loss = 0 # Placeholder
        ratio = 0 # Placeholder

        # ---------------- Problem 1.4.2: KL Divergence Policy Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.4.2 ###
        kl = (old_log_probs - log_probs).mean()  # or use true KL if possible
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean() + self.beta * kl
        ### END STUDENT SOLUTION - 1.4.2 ###
        
        # ---------------- Problem 1.1.1: PPO Clipped Surrogate Objective Loss ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.1 ###
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        #policy_loss = -torch.min(surr1, surr2).mean() # negative sign because we want gradient ascent to maximzie objective function
        
        total_loss = policy_loss
        ### END STUDENT SOLUTION - 1.1.1 ###
        
        
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)

        entropy_loss = 0 # Placeholder
        value_loss = 0 # Placeholder

        # ---------------- Problem 1.1.2: PPO Total Loss (Include Entropy Bonus and Value Loss) ----------------
        ### BEGIN STUDENT SOLUTION - 1.1.2 ###
        entropy_loss = -entropy.mean()  # negative sign because we want to maximize entropy  
        value_loss =  ((returns - values) ** 2).mean()
        total_loss = total_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss 
        ### END STUDENT SOLUTION - 1.1.2 ###

        # Stats
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
        
        return total_loss, {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(-entropy_loss.item()),
            "kl": float(approx_kl.item()),
            "clipfrac": float(clipfrac.item()),
        }
        
    def _prepare_batch(self, advantages, returns):
        """Collate the current rollout into a batch for the buffer"""
        obs = torch.stack([torch.as_tensor(t["obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        next_obs = torch.stack([torch.as_tensor(t["next_obs"], dtype=torch.float32) for t in self._curr_policy_rollout])
        actions = torch.stack([torch.as_tensor(t["action"], dtype=torch.float32) for t in self._curr_policy_rollout])
        log_probs = torch.tensor([t["log_prob"] for t in self._curr_policy_rollout], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in self._curr_policy_rollout], dtype=torch.float32)
        rewards = torch.tensor([t["reward"] for t in self._curr_policy_rollout], dtype=torch.float32)
        
        return {
            "obs": obs.to(self.device),
            "next_obs": next_obs.to(self.device),
            "actions": actions.to(self.device),
            "log_probs": log_probs.to(self.device),
            "rewards": rewards.to(self.device),
            "values": values.to(self.device),
            "dones": torch.tensor([t["done"] for t in self._curr_policy_rollout], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            "iteration": torch.full((len(self._curr_policy_rollout),), self._policy_iteration, dtype=torch.int32, device=self.device)
        }