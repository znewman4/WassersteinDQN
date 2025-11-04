# src/core/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml

@dataclass
class ExperimentCfg:
    name: str
    save_dir: str = "results/artifacts"
    log_dir: str = "results/logs"
    total_timesteps: int = 100_000
    eval_interval: int = 5_000
    seed: int = 42

@dataclass
class EnvCfg:
    name: str = "trading"
    data_path: Optional[str] = None
    dataset: Optional[str] = None
    window_size: int = 50
    max_steps: Optional[int] = None
    feature_cols: List[str] = field(default_factory=lambda: [])
    
    # Core financial params
    initial_cash: float = 1_000_000.0
    trade_size: float = 1.0
    commission_pct: float = 0.0
    slippage_pct: float = 0.0
    reward_mode: str = "pnl"

    # Backward-compatibility
    initial_balance: Optional[float] = None
    trading_fee: Optional[float] = None

    def __post_init__(self):
        if self.initial_balance is not None:
            self.initial_cash = self.initial_balance
        if self.trading_fee is not None:
            self.commission_pct = self.trading_fee


@dataclass
class AgentCfg:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    experiment: ExperimentCfg
    env: EnvCfg
    agent: AgentCfg


def load_yaml_config(path: str) -> Config:
    import os
    path = os.path.abspath(path)
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    agent_block = cfg_dict["agent"]
    # Automatically pack everything except "name" into params
    agent_name = agent_block.get("name")
    agent_params = {k: v for k, v in agent_block.items() if k != "name"}

    return Config(
        experiment=ExperimentCfg(**cfg_dict["experiment"]),
        env=EnvCfg(**cfg_dict["env"]),
        agent=AgentCfg(name=agent_name, params=agent_params),
    )

