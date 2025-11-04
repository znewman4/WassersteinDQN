from src.core.config import load_yaml_config
from src.core.train_runner import train_agent

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml_config(args.config)
    train_agent(cfg)
