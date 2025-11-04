# src/core/train_runner.py
import os, numpy as np, torch
from src.core.registry import AGENTS, ENVS, DATASETS
from src.core.config import Config
from src.evaluation.metrics import compute_pnl, compute_sharpe

def train_agent(cfg: Config):
    exp, env_cfg, agent_cfg = cfg.experiment, cfg.env, cfg.agent

    # 1. Load dataset
    if env_cfg.dataset:
        df = DATASETS.get(env_cfg.dataset)()
    else:
        import pandas as pd
        df = pd.read_parquet(env_cfg.data_path)

    # 2. Create environment
    EnvCls = ENVS.get(env_cfg.name)
    env = EnvCls(df, cfg)
    obs, _ = env.reset()

    # 3. Create agent
    AgentCls = AGENTS.get(agent_cfg.name)
    agent = AgentCls(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=agent_cfg.params,
    )

    total_timesteps = exp.total_timesteps
    eval_interval = exp.eval_interval
    rewards, ep_reward = [], 0
    os.makedirs(exp.save_dir, exist_ok=True)
    ckpt_path = os.path.join(exp.save_dir, f"{exp.name}.pt")

    for step in range(1, total_timesteps + 1):
        action = agent.act(obs)
        next_obs, reward, done, _, _ = env.step(action)
        agent.push(obs, action, reward, next_obs, done)
        agent.update()
        obs, ep_reward = next_obs, ep_reward + reward

        if done:
            rewards.append(ep_reward)
            obs, _ = env.reset()
            ep_reward = 0

        if step % eval_interval == 0:
            avg_r = np.mean(rewards[-10:]) if rewards else 0.0
            print(f"[Step {step}] AvgReward(10ep): {avg_r:.2f}")
            agent.save(ckpt_path)

    print(f"✅ Training finished. Saved model to: {ckpt_path}")
    return ckpt_path
