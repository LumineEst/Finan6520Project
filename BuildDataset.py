"""


Purpose:
This script performs Step 1 of the stock valuation project:
1. Downloads macroeconomic data from FRED.
2. Downloads stock price and volume data from Yahoo Finance.
3. Pulls basic company fundamentals from Yahoo Finance.
4. Transforms the data based on the project proposal.
5. Merges everything into one large DataFrame indexed by Date and Ticker.
6. Saves the final dataset to CSV.

Install needed packages first:
    pip install pandas numpy yfinance pandas_datareader openpyxl pyarrow


"""

import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

warnings.filterwarnings("ignore")

import urllib3
import requests

# Due to firewalls, I need to modify how my sessions are handled, using the below code.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================
# USER SETTINGS
# ==============================

START_DATE = "2019-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

TICKERS = [
    # Original companies
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "JPM", "XOM", "JNJ", "PG",
    "HD", "UNH", "V", "MA", "COST",
    "WMT", "BAC", "KO", "PEP", "DIS",

    # Technology and communication
    "AVGO", "AMD", "PLTR", "CRWD", "SNOW",

    # Industrials and automobiles
    "F", "GM", "CAT", "DE", "GE",

    # Healthcare
    "LLY", "PFE", "MRK", "CVS", "TMO",

    # Utilities
    "NEE", "DUK", "SO",

    # Energy and materials
    "SLB", "OXY", "NEM", "FCX",

    # Consumer and growth companies
    "LULU", "ETSY", "RIVN", "CAVA", "CELH",

    # Financial technology and aerospace
    "SOFI", "HOOD", "RKLB"
]

FORWARD_WINDOW_DAYS = 20
FRED_LAG_DAYS = 30
SEC_LAG_DAYS = 45
OUTPUT_CSV = "market_data.csv"
OUTPUT_EXCEL = "market_data.xlsx"


# ==============================
# FRED DATA DOWNLOAD
# ==============================

def download_fred_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Download macroeconomic data from FRED and return one cleaned DataFrame."""

    fred_symbols = {
        "10-Year Treasury Yield": "DGS10",
        "2-Year Treasury Yield": "DGS2",
        "CPI": "CPIAUCSL",
        "Federal Funds Rate": "FEDFUNDS",
        "Unemployment Rate": "UNRATE",
        "Financial Stress Index": "STLFSI4",
        "High Yield Credit Spread": "BAMLH0A0HYM2"
    }

    fred_frames = []

    for clean_name, fred_code in fred_symbols.items():
        print(f"Downloading FRED data: {clean_name} ({fred_code})")
        try:
            series = web.DataReader(fred_code, "fred", start_date, end_date)
            series = series.rename(columns={fred_code: clean_name})
            fred_frames.append(series)
        except Exception as e:
            print(f"WARNING: Could not download {clean_name}: {e}")

    if not fred_frames:
        raise RuntimeError("No FRED data downloaded. Check internet connection or pandas_datareader install.")

    fred = pd.concat(fred_frames, axis=1).sort_index()
    fred.index.name = "Date"

    # Convert CPI index into year-over-year inflation rate.
    # CPI is monthly, so 12-period percentage change gives YoY inflation.
    fred["CPI YoY Inflation"] = fred["CPI"].dropna().pct_change(12) * 100

    # Forward fill because some FRED series are monthly/weekly while stock data is daily.
    fred = fred.ffill()

    # Yield curve spread = 10-year yield minus 2-year yield.
    fred["Yield Curve Spread"] = fred["10-Year Treasury Yield"] - fred["2-Year Treasury Yield"]

    # Calculate delta from initial value for these macro variables.
    delta_cols = ["Federal Funds Rate", "Unemployment Rate", "Financial Stress Index"]
    for col in delta_cols:
        fred[f"{col} Delta"] = fred[col].diff(90)
    fred["Real 10Y Yield"] = fred["10-Year Treasury Yield"] - fred["CPI YoY Inflation"]

    # Lag FRED data to avoid accidentally using data before it was publicly available.
    # This is especially important for CPI, which is reported after the month ends.
    fred = fred.shift(freq=f"{FRED_LAG_DAYS}D")

    return fred


# ==============================
# STOCK PRICE DOWNLOAD
# ==============================

def download_stock_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adjusted close and volume for each stock from Yahoo Finance."""

    all_prices = []
    
    session = requests.Session()
    session.verify = False
    session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",})

    for ticker in tickers:
        print(f"Downloading stock prices: {ticker}")
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False,
                session=session
            )

            if data.empty:
                print(f"WARNING: No price data found for {ticker}")
                continue

            # Handle possible multi-index columns from yfinance.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()
            data["Ticker"] = ticker

            # Make sure Adj Close exists. If missing, use Close as a backup.
            if "Adj Close" not in data.columns and "Close" in data.columns:
                data["Adj Close"] = data["Close"]

            keep_cols = ["Date", "Ticker", "Adj Close", "Close", "Volume"]
            data = data[keep_cols]
            
            daily_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
            data["Month Volatility"] = daily_returns.rolling(21).std()
            data["Relative Volume"] = data["Volume"] / data["Volume"].rolling(21).mean()
            data["Month Momentum"] = np.log(data["Adj Close"] / data["Adj Close"].shift(21))
            data["Quarter Momentum"] = np.log(data["Adj Close"] / data["Adj Close"].shift(63))
            
            try:
                shares = yf.Ticker(ticker).info.get('sharesOutstanding', np.nan)
                data["Market Cap"] = data["Adj Close"] * shares
            except:
                data["Market Cap"] = np.nan

            all_prices.append(data)

        except Exception as e:
            print(f"WARNING: Could not download prices for {ticker}: {e}")

        time.sleep(0.25)

    if not all_prices:
        raise RuntimeError("No stock price data downloaded.")

    prices = pd.concat(all_prices, ignore_index=True)
    prices["Date"] = pd.to_datetime(prices["Date"])

    return prices


# ==============================
# FUNDAMENTALS DOWNLOAD
# ==============================

def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Pull current company fundamentals from Yahoo Finance."""

    rows = []
    sectors = {}

    for ticker in tickers:
        print(f"Downloading fundamentals: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            sectors[ticker] = stock.info.get("sector", "Unknown")
            
            # Fetch Quarterly Statements
            inc = stock.quarterly_financials.T if not stock.quarterly_financials.empty else pd.DataFrame()
            bs = stock.quarterly_balance_sheet.T if not stock.quarterly_balance_sheet.empty else pd.DataFrame()
            cf = stock.quarterly_cashflow.T if not stock.quarterly_cashflow.empty else pd.DataFrame()
            
            if inc.empty and bs.empty: continue
        
            statements = pd.concat([inc, bs, cf], axis=1)
            statements = statements.loc[:, ~statements.columns.duplicated()]
            
            def get_metric(df, keys):
                for k in keys:
                    if k in df.columns: return df[k]
                return np.nan
        
            rev = get_metric(statements, ["Total Revenue", "Operating Revenue"])
            netInc = get_metric(statements, ["Net Income"])
            assets = get_metric(statements, ["Total Assets"])
            equity = get_metric(statements, ["Stockholders Equity", "Total Stockholder Equity"])
            debt = get_metric(statements, ["Total Debt", "Long Term Debt And Capital Lease Obligation", 
                                           "Long Term Debt", "Current Debt"])
            if isinstance(debt, (float, int, np.floating)) and np.isnan(debt): debt = pd.Series(0.0, index=statements.index)
            else: debt = debt.fillna(0) 
            curAssets = get_metric(statements, ["Current Assets"])
            curLiab = get_metric(statements, ["Current Liabilities"])

            metricsDf = pd.DataFrame(index=statements.index)
            metricsDf["Ticker"] = ticker
            metricsDf["Operating Margin"] = get_metric(statements, ["Operating Income"]) / rev
            metricsDf["Gross Margin"] = get_metric(statements, ["Gross Profit"]) / rev
            metricsDf["ROE"] = netInc / equity
            metricsDf["ROA"] = netInc / assets
            metricsDf["Debt-to-Equity"] = debt / equity
            metricsDf["Current Ratio"] = curAssets / curLiab
            metricsDf["Free Cash Flow"] = get_metric(statements, ["Free Cash Flow"])
            metricsDf["EBITDA"] = get_metric(statements, ["EBITDA", "Normalized EBITDA"])
            metricsDf["Debt"] = debt
            metricsDf["Cash"] = get_metric(statements, ["Cash and Cash Equivalents", 
                                                        "Cash Cash Equivalents And Short Term Investments", 
                                                        "Cash Financial", "Cash"])

            metricsDf.index = pd.to_datetime(metricsDf.index) + pd.Timedelta(days=SEC_LAG_DAYS)
            metricsDf.index.name = "Date"

            rows.append(metricsDf.reset_index())

        except Exception as e:
            print(f"WARNING: Could not download fundamentals for {ticker}: {e}")

        time.sleep(0.5)

    if not rows:
        raise RuntimeError("No fundamentals downloaded.")

    fundamentals = pd.concat(rows, ignore_index=True)
    fundamentals = fundamentals.sort_values(["Ticker", "Date"])
    return fundamentals, sectors


# ==============================
# MERGE EVERYTHING
# ==============================

def build_master_dataset(prices: pd.DataFrame, fred: pd.DataFrame, fundamentals: pd.DataFrame, sectorMap: dict) -> pd.DataFrame:
    """Merge stock prices, macro data, and fundamentals into one modeling dataset."""

    prices = prices.copy()
    fred = fred.copy()
    fundamentals = fundamentals.copy()
    
    prices["Date"] = prices["Date"].astype("datetime64[ns]")
    fred.index = fred.index.astype("datetime64[ns]")
    fundamentals["Date"] = fundamentals["Date"].astype("datetime64[ns]")
    
    prices = prices.sort_values("Date")
    fred = fred.sort_values("Date")
    
    master = pd.merge_asof(prices, fred, left_on="Date", right_index=True, direction="backward")
    
    fundamentals = fundamentals.sort_values("Date")
    master = master.sort_values("Date")
    master = pd.merge_asof(master, fundamentals, on="Date", by="Ticker", direction="backward")

    master["Sector"] = master["Ticker"].map(sectorMap)
    
    ev = master["Market Cap"] + master["Debt"].fillna(0) - master["Cash"].fillna(0)
    master["EV/EBITDA"] = np.where(master["EBITDA"] > 0, ev / master["EBITDA"], np.nan)
    
    # ---------------------------------------------------------
    # HYBRID METRIC GENERATION
    # ---------------------------------------------------------

    # Macro-Fundamental Interaction (Credit Stress)
    # Heavily penalizes highly leveraged companies when credit spreads widen
    master["Credit Stress Exposure"] = master["Debt-to-Equity"] * master["High Yield Credit Spread"]

    # Relative Premium Spreads (FCF vs Risk-Free)
    master["FCF Yield"] = np.where(master["Market Cap"] > 0, master["Free Cash Flow"] / master["Market Cap"], np.nan)
    # Convert treasury yield from percentage (e.g., 4.5) to decimal (0.045) for fair comparison
    master["FCF Risk Premium"] = master["FCF Yield"] - (master["Real 10Y Yield"] / 100)

    # Rolling Macro Betas (Interest Rate Sensitivity)
    master["Daily Return"] = master.groupby("Ticker")["Adj Close"].pct_change()
    master["Yield Change"] = master.groupby("Ticker")["10-Year Treasury Yield"].diff()

    # Calculate 63-day rolling covariance and variance
    rolling_cov = master.groupby("Ticker").apply(lambda x: x["Daily Return"].rolling(63).cov(x["Yield Change"])).reset_index(level=0, drop=True)
    rolling_var = master.groupby("Ticker")["Yield Change"].transform(lambda x: x.rolling(63).var())

    # Apply your preferred variance floor (0.0001) to avoid division by zero
    rolling_var = rolling_var.where(rolling_var >= 1e-4, 1e-4).fillna(1e-4)
    master["10Y Yield Beta"] = rolling_cov / rolling_var

    # Clean up temporary calculation columns
    master = master.drop(columns=["Daily Return", "Yield Change", "FCF Yield"])

    # Drop rows without required modeling fields.
    required_cols = [
       "Adj Close", "Volume", "10-Year Treasury Yield", "2-Year Treasury Yield", "10Y Yield Beta",
       "Yield Curve Spread", "CPI YoY Inflation", "Federal Funds Rate Delta", "FCF Risk Premium",
       "Unemployment Rate Delta", "Financial Stress Index Delta", "Operating Margin",
       "Sector", "Real 10Y Yield", "Month Momentum", "Quarter Momentum", "Month Volatility",
       "High Yield Credit Spread", "Relative Volume", "Credit Stress Exposure" 
    ]

    existing_required = [col for col in required_cols if col in master.columns]
    master = master.dropna(subset=existing_required)
    
    master["Date"] = master["Date"].dt.strftime('%Y%m%d000000')

    # Sort and index the dataset as proposed: Date and Ticker.
    master = master.sort_values(["Date", "Ticker"])
    master = master.set_index(["Date", "Ticker"])

    return master


# ==============================
# MAIN PROGRAM
# ==============================

def main():
    print("Starting Step 1: Data Collection and Processing")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {', '.join(TICKERS)}")

    fred = download_fred_data(START_DATE, END_DATE)
    prices = download_stock_prices(TICKERS, START_DATE, END_DATE)
    fundamentals, sectors = download_fundamentals(TICKERS)

    master = build_master_dataset(prices, fred, fundamentals, sectors)

    print(f"\nFinal dataset shape: {master.shape}")
    print("Columns included:")
    print(list(master.columns))

    # Save files.
    master.to_csv(OUTPUT_CSV)
    print(f"Saved CSV: {OUTPUT_CSV}")

    try:
        # Excel has row limits, so save only if not too large.
        if len(master) <= 1_000_000:
            master.to_excel(OUTPUT_EXCEL)
            print(f"Saved Excel: {OUTPUT_EXCEL}")
        else:
            print("Excel save skipped because dataset is too large for Excel.")
    except Exception as e:
        print(f"Excel save skipped: {e}")

    print("\nStep 1 complete.")


if __name__ == "__main__":
    main()
