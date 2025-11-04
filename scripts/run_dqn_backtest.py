#scripts/run_dqn_backtest.py
import os
import torch
import pandas as pd
from src.utils.io import load_yaml_config
from src.env.trading_env import TradingEnv
from src.agents.dqn import DQNAgent
from src.evaluation.metrics import compute_pnl, compute_sharpe, compute_maxdd
from src.evaluation.plots import plot_equity_curve

def run_dqn_backtest(config_path: str):
    cfg = load_yaml_config(config_path)
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    exp_cfg = cfg["experiment"]

    # 1. Load dataset
    df = pd.read_parquet(env_cfg["data_path"])

    # 2. Init environment
    env = TradingEnv(df, cfg)
    obs, _ = env.reset()

    # 3. Init DQN agent & load weights
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(obs_dim, action_dim, agent_cfg)
    checkpoint_path = os.path.join(exp_cfg["save_dir"], "dqn.pt")
    agent.load(checkpoint_path)
    agent.epsilon = 0.0  # no exploration in eval

    # 4. Run backtest
    done = False
    rewards, equity_curve, trades = [], [], []
    while not done:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        equity_curve.append(info.get("equity", 0))
        if "trade" in info and info["trade"] is not None:
            trades.append(info["trade"])

    # 5. Compute metrics
    pnl = compute_pnl(equity_curve)
    sharpe = compute_sharpe(equity_curve)
    maxdd = compute_maxdd(equity_curve)
    print(f"✅ Backtest complete | PnL={pnl:.2f} | Sharpe={sharpe:.2f} | MaxDD={maxdd:.2f}")

    # 6. Plot & save
    plot_path = os.path.join("results/plots", f"{exp_cfg['name']}_equity.png")
    os.makedirs("results/plots", exist_ok=True)
    plot_equity_curve(equity_curve, trades, plot_path)
    print(f"📈 Saved equity curve to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/exp010_dqn_v1.yaml")
    args = parser.parse_args()

    run_dqn_backtest(args.config)
