"""
Adversarial defenses for cloud scheduling RL agent.

This module implements adversarial training and other defense mechanisms
to improve the robustness of the RL-based cloud scheduling agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import sys
import os
from collections import defaultdict

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scheduler.rl_model.agents.gin_agent.agent import GinAgent
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentObsTensor
from scheduler.rl_model.adversarial.attacks import AttackConfig, BaseAttack, AttackEnsemble


@dataclass
class DefenseConfig:
    """Configuration for adversarial defenses."""
    # Adversarial training
    adv_training_ratio: float = 0.5  # Fraction of training samples to be adversarial
    adv_training_epochs: int = 10  # Number of adversarial training epochs
    adv_lr: float = 1e-4  # Learning rate for adversarial training
    
    # Defense mechanisms
    use_ensemble_defense: bool = True  # Use ensemble of models for defense
    use_input_validation: bool = True  # Validate inputs before processing
    use_uncertainty_threshold: bool = True  # Use uncertainty threshold for decisions
    
    # Uncertainty threshold
    uncertainty_threshold: float = 0.8  # Threshold for uncertainty-based decisions
    confidence_threshold: float = 0.7  # Minimum confidence for action selection


class AdversarialTrainingDefense:
    """
    Adversarial training defense mechanism.
    
    This defense trains the agent on both clean and adversarial examples
    to improve robustness against attacks.
    """
    
    def __init__(self, config: DefenseConfig, attack_config: AttackConfig, device: torch.device):
        self.config = config
        self.attack_config = attack_config
        self.device = device
        self.attack_ensemble = None
    
    def train_adversarially(self, agent: GinAgent, clean_obs: List[GinAgentObsTensor], 
                          clean_actions: List[torch.Tensor], clean_rewards: List[torch.Tensor]) -> GinAgent:
        """
        Train the agent using adversarial training.
        
        Args:
            agent: The RL agent to train
            clean_obs: List of clean observations
            clean_actions: List of corresponding actions
            clean_rewards: List of corresponding rewards
        
        Returns:
            Adversarially trained agent
        """
        if self.attack_ensemble is None:
            self.attack_ensemble = AttackEnsemble(self.attack_config, agent, self.device)
        
        # Create adversarial examples
        adv_obs = []
        adv_actions = []
        adv_rewards = []
        
        for i, obs in enumerate(clean_obs):
            # Generate adversarial examples using ensemble
            attack_results = self.attack_ensemble.evaluate_all_attacks(obs, agent)
            
            # Select the most effective attack (highest loss)
            best_attack_obs = obs
            best_attack_name = "clean"
            max_loss = float('-inf')
            
            for attack_name, attack_obs in attack_results.items():
                if attack_name == "clean":
                    continue
                
                # Compute loss for this attack
                with torch.no_grad():
                    target_scores = agent(attack_obs)
                    target_probs = torch.softmax(target_scores, dim=-1)
                    entropy = -torch.sum(target_probs * torch.log(target_probs + 1e-8), dim=-1)
                    loss = entropy.mean().item()
                
                if loss > max_loss:
                    max_loss = loss
                    best_attack_obs = attack_obs
                    best_attack_name = attack_name
            
            # Add adversarial example
            adv_obs.append(best_attack_obs)
            adv_actions.append(clean_actions[i])
            adv_rewards.append(clean_rewards[i])
        
        # Combine clean and adversarial data
        all_obs = clean_obs + adv_obs
        all_actions = clean_actions + adv_actions
        all_rewards = clean_rewards + adv_rewards
        
        # Shuffle the combined dataset
        indices = list(range(len(all_obs)))
        random.shuffle(indices)
        
        shuffled_obs = [all_obs[i] for i in indices]
        shuffled_actions = [all_actions[i] for i in indices]
        shuffled_rewards = [all_rewards[i] for i in indices]
        
        # Train the agent on the combined dataset
        self._train_agent_on_data(agent, shuffled_obs, shuffled_actions, shuffled_rewards)
        
        return agent
    
    def _train_agent_on_data(self, agent: GinAgent, obs_list: List[GinAgentObsTensor], 
                            actions_list: List[torch.Tensor], rewards_list: List[torch.Tensor]):
        """Train the agent on the provided data."""
        optimizer = optim.Adam(agent.parameters(), lr=self.config.adv_lr)
        
        for epoch in range(self.config.adv_training_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Create batches
            batch_size = min(32, len(obs_list))
            for i in range(0, len(obs_list), batch_size):
                batch_obs = obs_list[i:i+batch_size]
                batch_actions = actions_list[i:i+batch_size]
                batch_rewards = rewards_list[i:i+batch_size]
                
                if len(batch_obs) == 0:
                    continue
                
                optimizer.zero_grad()
                
                # Compute loss for this batch
                batch_loss = 0.0
                for j, obs in enumerate(batch_obs):
                    if j >= len(batch_actions) or j >= len(batch_rewards):
                        continue
                    
                    # Get agent's action and value
                    action_scores = agent(obs)
                    action_probs = torch.softmax(action_scores, dim=-1)
                    
                    # Compute policy loss (negative log likelihood)
                    target_action = batch_actions[j]
                    if target_action < action_probs.shape[1]:
                        policy_loss = -torch.log(action_probs[0, target_action] + 1e-8)
                    else:
                        policy_loss = torch.tensor(0.0, device=self.device)
                    
                    # Compute value loss (MSE with actual reward)
                    value = agent.get_value(obs.unsqueeze(0) if obs.dim() == 1 else obs)
                    value_loss = (value - batch_rewards[j]) ** 2
                    
                    # Total loss
                    total_loss = policy_loss + 0.5 * value_loss
                    batch_loss += total_loss
                
                if len(batch_obs) > 0:
                    batch_loss = batch_loss / len(batch_obs)
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Adversarial training epoch {epoch+1}/{self.config.adv_training_epochs}, "
                      f"avg loss: {avg_loss:.4f}")


class EnsembleDefense:
    """
    Ensemble defense mechanism using multiple models.
    
    This defense uses an ensemble of models to make more robust decisions
    and detect potential adversarial inputs.
    """
    
    def __init__(self, config: DefenseConfig, num_models: int = 5, device: torch.device = None):
        self.config = config
        self.num_models = num_models
        self.device = device or torch.device('cpu')
        self.models = []
        self.attack_ensemble = None
    
    def train_ensemble(self, base_agent: GinAgent, training_data: List[Tuple[GinAgentObsTensor, torch.Tensor, torch.Tensor]]):
        """Train an ensemble of models with different initializations."""
        self.models = []
        
        for i in range(self.num_models):
            # Create a new model with different initialization
            model = GinAgent(self.device)
            
            # Train on different subsets of data
            subset_size = len(training_data) // self.num_models
            start_idx = i * subset_size
            end_idx = start_idx + subset_size if i < self.num_models - 1 else len(training_data)
            
            subset_data = training_data[start_idx:end_idx]
            
            if len(subset_data) > 0:
                obs_list = [item[0] for item in subset_data]
                actions_list = [item[1] for item in subset_data]
                rewards_list = [item[2] for item in subset_data]
                
                # Train this model
                self._train_single_model(model, obs_list, actions_list, rewards_list)
            
            self.models.append(model)
    
    def _train_single_model(self, model: GinAgent, obs_list: List[GinAgentObsTensor], 
                           actions_list: List[torch.Tensor], rewards_list: List[torch.Tensor]):
        """Train a single model on the provided data."""
        optimizer = optim.Adam(model.parameters(), lr=self.config.adv_lr)
        
        for epoch in range(5):  # Fewer epochs for ensemble training
            total_loss = 0.0
            num_batches = 0
            
            batch_size = min(16, len(obs_list))
            for i in range(0, len(obs_list), batch_size):
                batch_obs = obs_list[i:i+batch_size]
                batch_actions = actions_list[i:i+batch_size]
                batch_rewards = rewards_list[i:i+batch_size]
                
                if len(batch_obs) == 0:
                    continue
                
                optimizer.zero_grad()
                
                batch_loss = 0.0
                for j, obs in enumerate(batch_obs):
                    if j >= len(batch_actions) or j >= len(batch_rewards):
                        continue
                    
                    action_scores = model(obs)
                    action_probs = torch.softmax(action_scores, dim=-1)
                    
                    target_action = batch_actions[j]
                    if target_action < action_probs.shape[1]:
                        policy_loss = -torch.log(action_probs[0, target_action] + 1e-8)
                    else:
                        policy_loss = torch.tensor(0.0, device=self.device)
                    
                    value = model.get_value(obs.unsqueeze(0) if obs.dim() == 1 else obs)
                    value_loss = (value - batch_rewards[j]) ** 2
                    
                    total_loss = policy_loss + 0.5 * value_loss
                    batch_loss += total_loss
                
                if len(batch_obs) > 0:
                    batch_loss = batch_loss / len(batch_obs)
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                    num_batches += 1
    
    def predict(self, obs: GinAgentObsTensor) -> Tuple[torch.Tensor, float]:
        """
        Make a prediction using the ensemble.
        
        Returns:
            Tuple of (action_scores, confidence)
        """
        if not self.models:
            raise ValueError("No models in ensemble. Call train_ensemble first.")
        
        all_scores = []
        all_values = []
        
        for model in self.models:
            with torch.no_grad():
                scores = model(obs)
                value = model.get_value(obs.unsqueeze(0) if obs.dim() == 1 else obs)
                all_scores.append(scores)
                all_values.append(value)
        
        # Average the scores and values
        avg_scores = torch.stack(all_scores).mean(dim=0)
        avg_value = torch.stack(all_values).mean(dim=0)
        
        # Compute confidence as the agreement between models
        scores_tensor = torch.stack(all_scores)
        probs_tensor = torch.softmax(scores_tensor, dim=-1)
        
        # Confidence is the average probability of the most likely action
        max_probs, _ = torch.max(probs_tensor, dim=-1)
        confidence = max_probs.mean().item()
        
        return avg_scores, confidence
    
    def is_adversarial(self, obs: GinAgentObsTensor) -> bool:
        """
        Detect if an observation is likely adversarial.
        
        Returns:
            True if the observation is likely adversarial
        """
        if not self.models:
            return False
        
        # Get predictions from all models
        all_scores = []
        for model in self.models:
            with torch.no_grad():
                scores = model(obs)
                all_scores.append(scores)
        
        scores_tensor = torch.stack(all_scores)
        probs_tensor = torch.softmax(scores_tensor, dim=-1)
        
        # Compute disagreement between models
        mean_probs = probs_tensor.mean(dim=0)
        disagreement = torch.sum((probs_tensor - mean_probs) ** 2, dim=-1).mean().item()
        
        # High disagreement indicates potential adversarial input
        return disagreement > self.config.uncertainty_threshold


class InputValidationDefense:
    """
    Input validation defense mechanism.
    
    This defense validates inputs before processing to detect
    and filter out potentially adversarial inputs.
    """
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.feature_ranges = {}  # Will be populated during training
    
    def set_feature_ranges(self, training_obs: List[GinAgentObsTensor]):
        """Set valid feature ranges based on training data."""
        if not training_obs:
            return
        
        # Compute statistics for each feature
        task_lengths = torch.cat([obs.task_length for obs in training_obs])
        vm_speeds = torch.cat([obs.vm_speed for obs in training_obs])
        task_memory = torch.cat([obs.task_memory_req_mb for obs in training_obs])
        vm_memory = torch.cat([obs.vm_memory_mb for obs in training_obs])
        
        self.feature_ranges = {
            'task_length': (task_lengths.min().item(), task_lengths.max().item()),
            'vm_speed': (vm_speeds.min().item(), vm_speeds.max().item()),
            'task_memory': (task_memory.min().item(), task_memory.max().item()),
            'vm_memory': (vm_memory.min().item(), vm_memory.max().item()),
        }
    
    def validate_input(self, obs: GinAgentObsTensor) -> bool:
        """
        Validate if the input is within expected ranges.
        
        Returns:
            True if the input is valid
        """
        if not self.feature_ranges:
            return True  # No validation if ranges not set
        
        # Check task length
        if 'task_length' in self.feature_ranges:
            min_val, max_val = self.feature_ranges['task_length']
            if torch.any(obs.task_length < min_val) or torch.any(obs.task_length > max_val):
                return False
        
        # Check VM speed
        if 'vm_speed' in self.feature_ranges:
            min_val, max_val = self.feature_ranges['vm_speed']
            if torch.any(obs.vm_speed < min_val) or torch.any(obs.vm_speed > max_val):
                return False
        
        # Check memory values
        if 'task_memory' in self.feature_ranges:
            min_val, max_val = self.feature_ranges['task_memory']
            if torch.any(obs.task_memory_req_mb < min_val) or torch.any(obs.task_memory_req_mb > max_val):
                return False
        
        if 'vm_memory' in self.feature_ranges:
            min_val, max_val = self.feature_ranges['vm_memory']
            if torch.any(obs.vm_memory_mb < min_val) or torch.any(obs.vm_memory_mb > max_val):
                return False
        
        return True
    
    def sanitize_input(self, obs: GinAgentObsTensor) -> GinAgentObsTensor:
        """
        Sanitize the input by clipping values to valid ranges.
        
        Returns:
            Sanitized observation
        """
        if not self.feature_ranges:
            return obs
        
        new_obs = obs
        
        # Sanitize task length
        if 'task_length' in self.feature_ranges:
            min_val, max_val = self.feature_ranges['task_length']
            new_task_length = torch.clamp(obs.task_length, min_val, max_val)
            new_obs = GinAgentObsTensor(
                task_state_scheduled=new_obs.task_state_scheduled,
                task_state_ready=new_obs.task_state_ready,
                task_length=new_task_length,
                task_completion_time=new_obs.task_completion_time,
                task_memory_req_mb=new_obs.task_memory_req_mb,
                task_cpu_req_cores=new_obs.task_cpu_req_cores,
                vm_completion_time=new_obs.vm_completion_time,
                vm_speed=new_obs.vm_speed,
                vm_energy_rate=new_obs.vm_energy_rate,
                vm_memory_mb=new_obs.vm_memory_mb,
                vm_available_memory_mb=new_obs.vm_available_memory_mb,
                vm_used_memory_fraction=new_obs.vm_used_memory_fraction,
                vm_active_tasks_count=new_obs.vm_active_tasks_count,
                vm_next_release_time=new_obs.vm_next_release_time,
                vm_cpu_cores=new_obs.vm_cpu_cores,
                vm_available_cpu_cores=new_obs.vm_available_cpu_cores,
                vm_used_cpu_fraction_cores=new_obs.vm_used_cpu_fraction_cores,
                vm_next_core_release_time=new_obs.vm_next_core_release_time,
                compatibilities=new_obs.compatibilities,
                task_dependencies=new_obs.task_dependencies
            )
        
        # Similar sanitization for other features...
        # (Implementation would be similar for other features)
        
        return new_obs


class RobustAgent(nn.Module):
    """
    Robust agent that combines multiple defense mechanisms.
    """
    
    def __init__(self, base_agent: GinAgent, defense_config: DefenseConfig, 
                 attack_config: AttackConfig, device: torch.device):
        super().__init__()
        self.base_agent = base_agent
        self.defense_config = defense_config
        self.attack_config = attack_config
        self.device = device
        
        # Initialize defense mechanisms
        self.ensemble_defense = None
        self.input_validation = InputValidationDefense(defense_config)
        
        if defense_config.use_ensemble_defense:
            self.ensemble_defense = EnsembleDefense(defense_config, device=device)
    
    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        """Forward pass with defense mechanisms."""
        # Input validation
        if self.defense_config.use_input_validation:
            if not self.input_validation.validate_input(obs):
                # Sanitize the input
                obs = self.input_validation.sanitize_input(obs)
        
        # Use ensemble if available
        if self.ensemble_defense and self.ensemble_defense.models:
            scores, confidence = self.ensemble_defense.predict(obs)
            
            # Check confidence threshold
            if self.defense_config.use_uncertainty_threshold:
                if confidence < self.defense_config.confidence_threshold:
                    # Low confidence - use base agent as fallback
                    return self.base_agent(obs)
            
            return scores
        else:
            # Use base agent
            return self.base_agent(obs)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value with defense mechanisms."""
        if self.ensemble_defense and self.ensemble_defense.models:
            _, confidence = self.ensemble_defense.predict(obs)
            if confidence < self.defense_config.confidence_threshold:
                return self.base_agent.get_value(obs)
        
        return self.base_agent.get_value(obs)
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action and value with defense mechanisms."""
        return self.base_agent.get_action_and_value(obs, action)

