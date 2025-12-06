"""
FIXED: Fetch DAILY asset data for Indian markets
The previous version was accidentally creating monthly data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def fetch_daily_prices(config_path='config.yaml', save=True):
    """
    Fetch DAILY Indian asset price data from Yahoo Finance
    
    Returns:
    --------
    pd.DataFrame : DAILY price data
    """
    
    config = load_config(config_path)
    
    # Collect all assets
    all_assets = []
    for category in ['equities', 'bonds', 'commodities', 'international']:
        if category in config['assets']:
            all_assets.extend(config['assets'][category])
    
    all_assets = list(dict.fromkeys(all_assets))
    
    start = config['start_date']
    end = config['end_date']
    
    logger.info(f"Fetching DAILY data for {len(all_assets)} assets")
    logger.info(f"Date range: {start} to {end}")
    logger.info(f"Assets: {', '.join(all_assets)}")
    
    try:
        # Download DAILY data
        data = yf.download(
            all_assets, 
            start=start, 
            end=end,
            interval='1d',  # DAILY data
            group_by='ticker',
            progress=True,
            auto_adjust=True
        )
        
        logger.info(f"✓ Successfully downloaded DAILY data")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"  Frequency: {pd.infer_freq(data.index)} (should be 'B' for business daily)")
        logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"  Total days: {len(data)}")
        
        # Verify we got daily data
        days_expected = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if len(data) < days_expected * 0.5:  # Should have at least 50% of calendar days
            logger.warning(f"⚠️ Data might not be daily! Only {len(data)} rows for {days_expected} calendar days")
        
        if save:
            output_path = Path('data/raw/india_asset_prices_daily.parquet')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_parquet(output_path)
            logger.info(f"✓ Saved to {output_path}")
            
            # Save metadata
            metadata = {
                'fetch_date': datetime.now().isoformat(),
                'assets': all_assets,
                'start_date': start,
                'end_date': end,
                'frequency': 'daily',
                'n_rows': len(data),
                'n_assets': len(all_assets)
            }
            
            metadata_path = Path('data/raw/india_asset_prices_daily_metadata.yaml')
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
        
        return data
        
    except Exception as e:
        logger.error(f"✗ Failed to fetch asset data: {e}")
        raise


def compute_daily_returns(data, save=True):
    """
    Compute DAILY log returns from price data
    
    Parameters:
    -----------
    data : pd.DataFrame
        DAILY price data
        
    Returns:
    --------
    pd.DataFrame : DAILY log returns
    """
    
    logger.info(f"Computing DAILY log returns...")
    
    # Handle both single asset and multi-asset DataFrames
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    # Compute log returns
    returns = np.log(prices / prices.shift(1))
    
    # Remove first row (NaN)
    returns = returns.dropna()
    
    logger.info(f"✓ Computed DAILY returns")
    logger.info(f"  Shape: {returns.shape}")
    logger.info(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # Sanity checks
    logger.info("\nSanity Checks:")
    logger.info(f"  Mean daily return: {returns.mean().mean()*100:.4f}%")
    logger.info(f"  Mean annualized return (×252): {returns.mean().mean()*252*100:.2f}%")
    logger.info(f"  Daily volatility: {returns.std().mean()*100:.4f}%")
    logger.info(f"  Annualized volatility (×√252): {returns.std().mean()*np.sqrt(252)*100:.2f}%")
    
    # Check COVID crash specifically
    if '2020-03' in returns.index.to_series().astype(str).values:
        march_2020 = returns.loc['2020-03']
        logger.info(f"\nMarch 2020 Check:")
        logger.info(f"  Days of data: {len(march_2020)}")
        logger.info(f"  Worst daily return: {march_2020.min().min()*100:.2f}%")
        logger.info(f"  Cumulative return: {(np.exp(march_2020.sum().mean()) - 1)*100:.2f}%")
    
    if save:
        output_path = Path('data/processed/india_asset_returns_daily.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(output_path)
        logger.info(f"\n✓ Saved DAILY returns to {output_path}")
    
    return returns


def create_monthly_returns(daily_returns, save=True):
    """
    Aggregate daily returns to monthly for regime detection
    
    Parameters:
    -----------
    daily_returns : pd.DataFrame
        Daily log returns
        
    Returns:
    --------
    pd.DataFrame : Monthly log returns
    """
    
    logger.info("Creating MONTHLY returns from daily data...")
    
    # For log returns, monthly return = sum of daily log returns
    monthly_returns = daily_returns.resample('M').sum()
    
    logger.info(f"✓ Created monthly returns")
    logger.info(f"  Shape: {monthly_returns.shape}")
    logger.info(f"  Date range: {monthly_returns.index[0]} to {monthly_returns.index[-1]}")
    
    if save:
        output_path = Path('data/processed/india_asset_returns_monthly.parquet')
        monthly_returns.to_parquet(output_path)
        logger.info(f"✓ Saved to {output_path}")
    
    return monthly_returns


def get_asset_summary(data):
    """Generate summary statistics"""
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data.xs('Close', level=1, axis=1)
    else:
        prices = data['Close']
    
    summary = pd.DataFrame({
        'First Date': prices.apply(lambda x: x.first_valid_index()),
        'Last Date': prices.apply(lambda x: x.last_valid_index()),
        'Trading Days': prices.count(),
        'Missing Values': prices.isnull().sum(),
        'Missing %': (prices.isnull().sum() / len(prices) * 100).round(2),
        'First Price': prices.apply(lambda x: x.iloc[0] if not x.isnull().all() else None),
        'Last Price': prices.apply(lambda x: x.iloc[-1] if not x.isnull().all() else None),
    })
    
    return summary


if __name__ == "__main__":
    
    logger.info("="*70)
    logger.info("FETCHING DAILY INDIAN ASSET DATA (FIXED VERSION)")
    logger.info("="*70)
    
    # Fetch DAILY prices
    logger.info("\n[Step 1] Fetching DAILY Prices")
    data = fetch_daily_prices()
    
    # Generate summary
    logger.info("\n" + "="*70)
    logger.info("ASSET SUMMARY")
    logger.info("="*70)
    summary = get_asset_summary(data)
    print(summary)
    
    # Compute DAILY returns
    logger.info("\n[Step 2] Computing DAILY Returns")
    daily_returns = compute_daily_returns(data)
    
    # Create MONTHLY returns for regime detection
    logger.info("\n[Step 3] Creating MONTHLY Returns")
    monthly_returns = create_monthly_returns(daily_returns)
    
    logger.info("\n" + "="*70)
    logger.info("✓ DATA FETCH COMPLETE")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    logger.info("  - data/raw/india_asset_prices_daily.parquet (DAILY prices)")
    logger.info("  - data/processed/india_asset_returns_daily.parquet (DAILY returns)")
    logger.info("  - data/processed/india_asset_returns_monthly.parquet (MONTHLY returns)")
    logger.info("\nNext steps:")
    logger.info("  1. Use MONTHLY returns for regime detection (HMM)")
    logger.info("  2. Use DAILY returns for backtesting and portfolio analysis")