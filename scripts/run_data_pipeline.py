
import logging
import argparse
from pathlib import Path
import pandas as pd

from src.utils.io import load_config
from src.data_pipeline.ingest import seed_ohlcv, append_new_ohlcv
from src.data_pipeline.clean import clean_and_resample, save_clean
from src.data_pipeline.features import add_basic_features  # ✅ new import

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("run_data_pipeline")

FREQ_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1H", "4h": "4H", "1d": "1D"
}

def main(mode: str):
    # 1. Load config
    cfg = load_config("configs/data.yaml")
    ex_cfg = cfg["data"]

    symbol = ex_cfg["symbol"].replace("/", "").lower()
    timeframe = ex_cfg["timeframe"]

    raw_dir = Path(ex_cfg["paths"]["raw"])
    interim_dir = Path(ex_cfg["paths"]["interim"])
    processed_dir = Path(ex_cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / f"{symbol}_{timeframe}.parquet"
    interim_file = interim_dir / f"{symbol}_{timeframe}.parquet"
    processed_file = processed_dir / f"{symbol}_{timeframe}.parquet"

    # -------------------
    # Fetch or Local mode
    # -------------------
    if mode == "fetch":
        if not raw_file.exists():
            logger.info("Seeding dataset from %s to %s",
                        ex_cfg["start_date"], ex_cfg.get("end_date", "now"))
            df = seed_ohlcv(cfg, raw_file)
        else:
            logger.info("Appending new data to %s", raw_file)
            df = append_new_ohlcv(cfg, raw_file)

        # Clean
        pandas_freq = FREQ_MAP.get(timeframe, timeframe)
        logger.info("Cleaning dataset with pandas freq=%s", pandas_freq)
        df_clean = clean_and_resample(df, freq=pandas_freq)

        # Save cleaned to interim
        save_clean(df_clean, interim_file)
        df_source = df_clean

    else:  # mode == "local"
        logger.info("📂 Using local interim parquet: %s", interim_file)
        if not interim_file.exists():
            raise FileNotFoundError(f"No interim file found at {interim_file}")
        df_source = pd.read_parquet(interim_file)

    # -------------------
    # Add features + save
    # -------------------
    logger.info("Adding basic features...")
    df_feat = add_basic_features(df_source)
    df_feat.to_parquet(processed_file)
    logger.info("✅ Saved feature-enriched data to %s (%d rows)", processed_file, len(df_feat))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fetch", "local"], default="local",
                        help="fetch = pull new data from CCXT, local = reuse existing parquet")
    args = parser.parse_args()
    main(args.mode)
