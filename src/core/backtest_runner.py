# src/core/backtest_runner.py
import os
from src.core.registry import AGENTS, ENVS, DATASETS
from src.evaluation.metrics import compute_pnl, compute_sharpe, compute_maxdd

def backtest_agent(cfg, checkpoint_path: str):
    exp, env_cfg, agent_cfg = cfg.experiment, cfg.env, cfg.agent

    # 1. Load dataset
    if env_cfg.dataset:
        df = DATASETS.get(env_cfg.dataset)()
    else:
        import pandas as pd
        df = pd.read_parquet(env_cfg.data_path)

    # 2. Env + agent setup
    EnvCls = ENVS.get(env_cfg.name)
    env = EnvCls(df, cfg)
    obs, _ = env.reset()

    AgentCls = AGENTS.get(agent_cfg.name)
    agent = AgentCls(env.observation_space.shape[0], env.action_space.n, agent_cfg.params)
    agent.load(checkpoint_path)
    if hasattr(agent, "epsilon"):  # disable exploration
        agent.epsilon = 0.0

    # 3. Evaluation loop
    done, equity, trades = False, [], []
    while not done:
        action = agent.act(obs)
        obs, reward, done, _, info = env.step(action)
        equity.append(info.get("equity", 0))
        if info.get("trade"): trades.append(info["trade"])

    pnl, sharpe, maxdd = (
        compute_pnl(equity),
        compute_sharpe(equity),
        compute_maxdd(equity),
    )

    print(f"✅ Backtest complete | PnL={pnl:.2f} | Sharpe={sharpe:.3f} | MaxDD={maxdd:.2f}")
    return {"pnl": pnl, "sharpe": sharpe, "maxdd": maxdd}
