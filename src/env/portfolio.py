#src/env/portfolio.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    initial_cash: float = 10_000.0
    trade_size: float = 1.0
    allow_short: bool = True
    max_position: Optional[float] = None  # cap abs(position) if set
    commission_pct: float = 0.0           # per side (e.g., 0.0005 = 5 bps)
    slippage_pct: float = 0.0             # price impact fraction

    # reward shaping
    sharpe_alpha: float = 0.0
    sharpe_lookback: int = 20
    sharpe_annualizer: float = 1.0
    dd_beta: float = 0.0


class Portfolio:
    """Self-contained portfolio & reward calculator."""
    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg
        self.reset()

    # ---------- lifecycle ----------
    def reset(self):
        self.cash: float = self.cfg.initial_cash
        self.position: float = 0.0
        self.equity: float = self.cash
        self.peak_equity: float = self.equity
        self.equity_history: List[float] = [self.equity]
        self.returns_history: List[float] = []
        self.trades: List[Tuple[int, int, float, float]] = []  # (step, action, px, size)

    # ---------- accounting ----------
    def mark_to_market(self, price: float) -> float:
        self.equity = self.cash + self.position * price
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_history.append(self.equity)
        return self.equity

    def _within_bounds(self, new_pos: float) -> bool:
        if self.cfg.max_position is None:
            return True
        return abs(new_pos) <= self.cfg.max_position

    def _exec_price(self, price: float, action: int) -> float:
        if action == 1:   # buy
            return price * (1.0 + self.cfg.slippage_pct)
        if action == -1:   # sell
            return price * (1.0 - self.cfg.slippage_pct)
        return price

    def apply_action(self, step: int, action: int, price: float) -> None:
        """
        Mutates cash/position with commission & slippage; records trade.
        """
        if action == 0:
            return

        px = self._exec_price(price, action)
        size = self.cfg.trade_size


        if action == 1:  # BUY
            cost = px * size
            fee = cost * self.cfg.commission_pct
            new_pos = self.position + size
            within = self._within_bounds(new_pos)
            cash_ok = self.cash >= cost + fee

            if within and cash_ok:
                self.position = new_pos
                self.cash -= (cost + fee)
                self.trades.append((step, action, px, size))

        elif action == 2:  # SELL
            proceeds = px * size
            fee = proceeds * self.cfg.commission_pct
            new_pos = self.position - size
            if (self.cfg.allow_short and self._within_bounds(new_pos)) or (not self.cfg.allow_short and self.position >= size):
                self.position = new_pos
                self.cash += (proceeds - fee)
                self.trades.append((step, action, px, size))

    # ---------- reward ----------
    def reward(
        self,
        prev_equity: float,
        new_equity: float,
    ) -> float:
        base = (new_equity - prev_equity) / prev_equity

        # step return for Sharpe
        step_ret = 0.0 if prev_equity <= 0 else (new_equity - prev_equity) / prev_equity
        self.returns_history.append(step_ret)

        shaped = base
        sharpe_term = 0.0
        if self.cfg.sharpe_alpha > 0 and len(self.returns_history) >= self.cfg.sharpe_lookback:
            r = np.array(self.returns_history[-self.cfg.sharpe_lookback:])
            std = r.std(ddof=1)
            if std > 0:
                sharpe = (r.mean() / std) * self.cfg.sharpe_annualizer
                sharpe_term = self.cfg.sharpe_alpha * float(sharpe)
                shaped += sharpe_term

        dd_term = 0.0
        if self.cfg.dd_beta > 0 and self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            dd_term = self.cfg.dd_beta * float(drawdown)
            shaped -= dd_term

        # expose components (optional)
        self._last_reward_components = {
            "base": float(base),
            "sharpe_term": float(sharpe_term),
            "drawdown_term": float(dd_term),
        }
        return float(shaped * 10_000.0) # scale up for learning stability

    @property
    def last_reward_components(self):
        return getattr(self, "_last_reward_components", {"base": 0.0, "sharpe_term": 0.0, "drawdown_term": 0.0})
