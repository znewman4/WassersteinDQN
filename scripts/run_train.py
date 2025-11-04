from src.core.config import load_yaml_config
from src.core.train_runner import train_agent

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="experiments/exp010_dqn_v1.yaml",   # ✅ default config here
        help="Path to YAML config file (default: experiments/exp010_dqn_v1.yaml)",
    )    
    args = p.parse_args()
    cfg = load_yaml_config(args.config)
    train_agent(cfg)
