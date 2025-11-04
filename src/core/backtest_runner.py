# src/core/backtest_runner.py
import os
import pandas as pd
from src.core.registry import AGENTS, ENVS, DATASETS
from src.evaluation.metrics import compute_pnl, compute_sharpe, compute_maxdd
from src.evaluation.plots import plot_equity_curve
from src.env import trading_env
from src.agents import dqn          # ensures AGENTS register runs


def backtest_agent(cfg, checkpoint_path: str, df: pd.DataFrame = None):
    """
    Run backtest using a trained agent and environment config.
    Produces metrics summary and labelled equity curve plot.
    """
    exp, env_cfg, agent_cfg = cfg.experiment, cfg.env, cfg.agent

    # 1️⃣ Load dataset
    if df is None:
        if env_cfg.dataset:
            df = DATASETS.get(env_cfg.dataset)()
        else:
            df = pd.read_parquet(env_cfg.data_path)

    # 2️⃣ Create environment + agent
    EnvCls = ENVS.get(env_cfg.name)
    env = EnvCls(df, cfg)
    obs, _ = env.reset()

    AgentCls = AGENTS.get(agent_cfg.name)
    agent = AgentCls(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        cfg=agent_cfg.params,
    )
    agent.load(checkpoint_path)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0  # disable exploration during backtest

    # 3️⃣ Run simulation loop
    done, equity, trades = False, [], []
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        equity.append(info.get("equity", env.portfolio.equity))

        # record structured trade info if portfolio has trades
        if env.portfolio.trades:
            step, act, px, sz = env.portfolio.trades[-1]
            trades.append({
                "step": step,
                "price": px,
                "size": sz,
                "type": "buy" if act == 1 else "sell" if act == -1 else "hold",
                "equity": env.portfolio.equity
            })

    # 4️⃣ Compute metrics
    pnl = compute_pnl(equity)
    sharpe = compute_sharpe(equity)
    maxdd = compute_maxdd(equity)

    metrics = {
        "pnl": pnl,
        "sharpe": sharpe,
        "maxdd": maxdd,
        "total_reward": total_reward,
        "num_trades": len(env.portfolio.trades),
    }

    print(
        f"✅ Backtest complete | PnL={pnl:.2f} | Sharpe={sharpe:.3f} "
        f"| MaxDD={maxdd:.2f} | Trades={len(env.portfolio.trades)}"
    )

    # 5️⃣ Create output directories
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # 6️⃣ Save metrics summary
    metrics_path = os.path.join("results/logs", f"{exp.name}_backtest_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"📊 Metrics saved to: {metrics_path}")

    # 7️⃣ Plot equity curve
    plot_path = os.path.join("results/plots", f"{exp.name}_equity_curve.png")
    plot_equity_curve(equity, trades, plot_path)
    print(f"📈 Equity curve saved to: {plot_path}")

    return metrics
