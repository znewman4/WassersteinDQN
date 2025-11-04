# src/env/trading_env.py
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.core.registry import ENVS
from .portfolio import Portfolio, PortfolioConfig
from .render_utils import EpisodeRenderer


def _read_config(cfg: Dict[str, Any]):
    """Support both dataclass Config and plain dicts."""
    if hasattr(cfg, "env"):
        env_cfg = vars(cfg.env)
        reward_obj = getattr(cfg, "reward", {})
        reward_cfg = vars(reward_obj) if hasattr(reward_obj, "__dict__") else reward_obj
    else:
        env_cfg = cfg.get("env", {})
        reward_cfg = cfg.get("reward", {})

    window_size = env_cfg.get("window_size", 10)
    max_steps = env_cfg.get("max_steps", None)

    p_cfg = PortfolioConfig(
        initial_cash=env_cfg.get("initial_cash", 1_000_000.0),
        trade_size=env_cfg.get("trade_size", 1.0),
        allow_short=env_cfg.get("allow_short", True),
        max_position=env_cfg.get("max_position", None),
        commission_pct=env_cfg.get("commission_pct", 0.0),
        slippage_pct=env_cfg.get("slippage_pct", 0.0),
        sharpe_alpha=reward_cfg.get("sharpe_alpha", 0.0),
        sharpe_lookback=reward_cfg.get("sharpe_lookback", 20),
        sharpe_annualizer=reward_cfg.get("sharpe_annualizer", 1.0),
        dd_beta=reward_cfg.get("dd_beta", 0.0),
    )
    return window_size, max_steps, env_cfg, p_cfg


@ENVS.register("trading")
class TradingEnv(gym.Env):
    """
    Gymnasium trading environment.
    Actions: 0=hold, 1=buy(+size), -1=sell(-size)
    Observation: last N closes + optional extra features
    Reward: Δequity + sharpe_alpha*Sharpe - dd_beta*drawdown
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        super().__init__()
        assert "Close" in df.columns, "DataFrame must contain a 'Close' column."

        self.df = df.reset_index(drop=True).copy()
        self.window_size, self.max_steps, self._env_cfg_raw, self.port_cfg = _read_config(config)
        self.portfolio = Portfolio(self.port_cfg)
        self.renderer = EpisodeRenderer()

        # optional configurable feature list
        self.feature_cols = self._env_cfg_raw.get("feature_cols", [
            "wasserstein_smooth_250", "wasserstein_smooth_1000",
            "momentum_sign", "ema_signal", "rsi_signal", "volatility"
        ])

        # spaces
        self.action_space = spaces.Discrete(3)

        # derive observation shape dynamically
        self.current_step = self.window_size
        obs_sample = self._get_obs()
        obs_shape = obs_sample.shape if isinstance(obs_sample, np.ndarray) else (len(obs_sample),)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        # state
        self.current_step: int = 0

    # ---- API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.portfolio.reset()
        self.portfolio.mark_to_market(float(self.df["Close"].iloc[self.current_step]))
        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action: int):
        px_now = float(self.df["Close"].iloc[self.current_step])
        prev_equity = self.portfolio.equity

        self.portfolio.apply_action(self.current_step, action, px_now)
        self.current_step += 1

        px_next = float(self.df["Close"].iloc[self.current_step])
        new_equity = self.portfolio.mark_to_market(px_next)
        reward = self.portfolio.reward(prev_equity, new_equity)

        terminated = self.current_step >= (len(self.df) - 1)
        truncated = False
        if self.max_steps is not None:
            truncated = (self.current_step - self.window_size) >= self.max_steps

        obs = self._get_obs()
        info = self._info()
        return obs, reward, terminated, truncated, info

    def render(self):
        closes = self.df["Close"].values
        self.renderer.render(
            closes=closes,
            cur_step=self.current_step,
            window_offset=self.window_size,
            equity_history=self.portfolio.equity_history,
            trades=self.portfolio.trades,
        )

    # ---- helpers ----
    def _get_obs(self) -> np.ndarray:
        s = self.current_step - self.window_size
        e = self.current_step

        # price window normalized to last close
        price_window = self.df["Close"].iloc[s:e].values.astype(np.float32)
        price_window = price_window / price_window[-1] - 1.0

        # dynamic feature inclusion
        extra_feats = []
        for col in self.feature_cols:
            if col in self.df.columns:
                val = float(self.df[col].iloc[e - 1])
                if np.isnan(val):
                    val = 0.0
                extra_feats.append(val)

        obs = np.concatenate([price_window, np.array(extra_feats, dtype=np.float32)])
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return obs.astype(np.float32)

    def _info(self) -> dict:
        row = self.df.iloc[self.current_step]
        row_lower = {k.lower(): v for k, v in row.items()}
        return {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "position": self.portfolio.position,
            "reward_components": self.portfolio.last_reward_components,
            "step": self.current_step,
            "momentum_sign": float(row_lower.get("momentum_sign", 0)),
            "ema_signal": float(row_lower.get("ema_signal", 0)),
            "rsi_signal": float(row_lower.get("rsi_signal", 0)),
            "volatility": float(row_lower.get("volatility", 0)),
        }
