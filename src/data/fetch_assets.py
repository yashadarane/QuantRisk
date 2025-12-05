"""
Fetches Indian equity, ETF, and forex data from Yahoo Finance
Robust to single- and multi-ticker downloads and handles adjusted vs non-adjusted closes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import logging

#set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    cfg_path = Path(config_path)
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _safe_ticker_list(config):
    #flatten and deduplicate tickers from config dict
    all_assets = []
    for category in ['equities', 'bonds', 'commodities', 'international']:
        if category in config.get('assets', {}):
            all_assets.extend(config['assets'][category])
    #preserve order, remove duplicates and empty strings
    return [t for t in list(dict.fromkeys(all_assets)) if t]


def _select_price_column(df, prefer_adjusted=True):
    if prefer_adjusted:
        if 'Adj Close' in df.columns:
            return df['Adj Close']
        if 'Close' in df.columns:
            return df['Close']
    else:
        if 'Close' in df.columns:
            return df['Close']
        if 'Adj Close' in df.columns:
            return df['Adj Close']
    # fallback: try first numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return df[numeric_cols[0]]
    raise ValueError("No usable price column found in dataframe.")


def fetch_india_assets(config_path='config.yaml', save=True, prefer_adjusted=True):
    """
    Fetch Indian asset price data from Yahoo Finance

    Returns:
        pd.DataFrame: Price data with MultiIndex columns if multiple tickers,
                      or a single-ticker DataFrame
    """
    config = load_config(config_path)
    tickers = _safe_ticker_list(config)
    if not tickers:
        raise ValueError("No tickers found in config['assets'].")

    start = config.get('start_date', '2005-01-01')
    end = config.get('end_date', None)

    logger.info(f"Fetching data for {len(tickers)} assets from {start} to {end}")
    logger.info(f"Assets: {', '.join(tickers)}")


    data = yf.download(
        tickers,
        start=start,
        end=end,
        group_by='ticker',
        progress=False,
        auto_adjust=False,  
        threads=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        logger.info("Multi-ticker download detected (MultiIndex columns).")
    else:
        logger.info("Single-ticker or flat dataframe returned by yfinance.")

    price_df = pd.DataFrame(index=data.index.unique()).sort_index()

    missing_tickers = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[0]:
                df_t = data[t].copy()
            else:
                df_t = data.copy()
            if 'Adj Close' in df_t.columns:
                prices = df_t['Adj Close']
            elif 'Close' in df_t.columns:
                prices = df_t['Close']
            else:
                prices = _select_price_column(df_t, prefer_adjusted=True)

            prices = prices.dropna(how='all')
            prices.index = pd.to_datetime(prices.index)
            prices = prices.sort_index()
            price_df = price_df.join(prices.rename(t), how='outer')
        except Exception as e:
            logger.warning(f"Failed to extract prices for {t}: {e}")
            missing_tickers.append(t)

    if missing_tickers:
        logger.warning(f"No usable data for tickers: {missing_tickers}")

    #basic sanity checks
    if price_df.empty or price_df.dropna(how='all').empty:
        raise RuntimeError("Downloaded data contains no valid price series.")

    non_null = price_df.dropna(how='all', axis=1)
    logger.info(f"✓ Price frame shape: {price_df.shape}")
    logger.info(f"✓ Date range: {price_df.index.min().date()} -> {price_df.index.max().date()}")
    logger.info(f"✓ Number of tickers with data: {len(non_null.columns)}")

    # Optionally save raw DataFrame (wide format)
    if save:
        out_raw = Path('data/raw/india_asset_prices_wide.parquet')
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        # write parquet (requires pyarrow or fastparquet)
        price_df.to_parquet(out_raw)
        logger.info(f"✓ Saved raw wide price table to {out_raw}")

        # metadata
        metadata = {
            'fetch_date': datetime.now().isoformat(),
            'requested_assets': tickers,
            'fetched_assets': list(non_null.columns),
            'start_date': start,
            'end_date': end,
            'n_rows': len(price_df),
            'n_assets': len(non_null.columns)
        }
        meta_path = Path('data/raw/india_asset_prices_metadata.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(metadata, f)
        logger.info(f"✓ Saved metadata to {meta_path}")

    return price_df


def compute_returns(price_df, return_type='log', freq=None, save=True):
    """
    Compute returns from the wide price DataFrame.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide DataFrame (index = dates, columns = tickers)
    return_type : {'log', 'simple'}
        Log returns or simple pct-change
    freq : str or None
        If provided (e.g., 'M'), resample prices to that frequency first (using last observation).
    save : bool
        Save processed returns to parquet.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame of returns
    """
    logger.info(f"Computing {return_type} returns (freq={freq})...")

    df = price_df.copy()

    # Optionally resample to monthly/weekly (use last observed price in period)
    if freq is not None:
        df = df.resample(freq).last()

    # Compute returns
    if return_type == 'log':
        returns = np.log(df / df.shift(1))
    elif return_type == 'simple':
        returns = df.pct_change()
    else:
        raise ValueError("return_type must be 'log' or 'simple'")

    returns = returns.dropna(how='all').dropna(axis=1, how='all')  # drop all-NaN rows/cols
    logger.info(f"✓ Computed returns. Shape: {returns.shape}")

    if save:
        out_path = Path(f'data/processed/india_asset_returns_{return_type}.parquet')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(out_path)
        logger.info(f"✓ Saved returns to {out_path}")

    return returns


def get_asset_summary(price_df):
    """
    Generate summary statistics for fetched assets (wide price table).
    """
    summary = pd.DataFrame(index=price_df.columns)
    summary['First Date'] = price_df.apply(lambda s: s.first_valid_index())
    summary['Last Date'] = price_df.apply(lambda s: s.last_valid_index())
    summary['Missing Values'] = price_df.isnull().sum()
    summary['Missing %'] = (price_df.isnull().sum() / len(price_df) * 100).round(2)
    summary['First Price'] = price_df.apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan)
    summary['Last Price'] = price_df.apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
    return summary


if __name__ == "__main__":
    # Quick run
    logger.info("=" * 60)
    logger.info("FETCHING INDIAN ASSET DATA")
    logger.info("=" * 60)
    data = fetch_india_assets()

    logger.info("\n" + "=" * 60)
    logger.info("ASSET SUMMARY")
    logger.info("=" * 60)
    print(get_asset_summary(data))

    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING RETURNS (monthly, log)")
    logger.info("=" * 60)
    returns = compute_returns(data, return_type='log', freq='M')

    logger.info("\n" + "=" * 60)
    logger.info("✓ DATA FETCH COMPLETE")
    logger.info("=" * 60)
