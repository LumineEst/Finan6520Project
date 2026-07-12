import yfinance as yf
import pandas as pd

# 1. Define the asset dictionary with clean, descriptive labels
index_map = {
    "^GSPC": "sp500",# Tracks the 500 largest U.S. companies; main benchmark for U.S. stock market performance.
    "^DJI": "dow_jones"#Tracks 30 major U.S. companies; often used as a headline market indicator.
    ,"^IXIC": "nasdaq",#Tech‑heavy U.S. index; includes many growth and technology stocks.
    "^RUT": "russell2000",#Tracks 2,000 small‑cap U.S. companies; used to measure small‑cap performance.
    "^VIX": "vix",#Volatility index; measures expected market fear/uncertainty.
    "^FTSE": "ftse100",#Major U.K. stock index; tracks 100 large British companies
    "^N225": "nikkei225",#Japan’s main stock index; tracks 225 large Japanese companies.
    "^STOXX50E": "eurostoxx50"#Tracks 50 major companies across Europe; benchmark for European markets.
    ,"GC=F": "gold",
    "SI=F": "silver",
    "PL=F": "platinum",
    "PA=F": "palladium",
    "CL=F": "crude_oil_wti",
    "BZ=F": "crude_oil_brent",
    "NG=F": "natural_gas",
    "HO=F": "heating_oil",
    "RB=F": "gasoline",
    "HG=F": "copper",
    "ALI=F": "aluminum",
    "ZNC=F": "zinc",
    "ZC=F": "corn",
    "ZW=F": "wheat",
    "ZS=F": "soybeans",
    "KC=F": "coffee",
    "SB=F": "sugar",
    "CT=F": "cotton",
    "CC=F": "cocoa",
    "LE=F": "live_cattle",
    "GF=F": "feeder_cattle",
    "HE=F": "lean_hogs"
}

# 2. Download all tickers simultaneously to optimize speed
tickers = list(index_map.keys())
raw_data = yf.download(tickers, start="2020-01-01", end="2026-07-10")

# 3. Extract Close prices (or 'Adj Close') and rename columns neatly
# This strips the messy '^' symbols and applies your preferred labels
clean_df = raw_data['Close'].rename(columns=index_map)

# 4. Clean up the index formatting for neat tabular viewing
clean_df.index = pd.to_datetime(clean_df.index).date
clean_df.index.name = 'Date'

# --- NEW ADDITIONS FOR PERCENT CHANGES & MOVEMENT COMPARISON ---

# 4b. Forward-fill missing entries so global holiday calendar mismatches do not break math
clean_df = clean_df.ffill()

# 4c. Calculate isolated Day-to-Day Percent Change for each index
daily_pct_change = clean_df.pct_change()

# 4d. Calculate Cumulative Percent Change (Base-Index Normalization)
# Normalizes all indices to start at exactly 0.0% on day one to compare movements neatly
cumulative_movements = (clean_df / clean_df.iloc[0] - 1) * 100

# --- EXCEL WORKBOOK EXPORT ---

with pd.ExcelWriter("global_indices_report.xlsx", engine="openpyxl") as writer:
    clean_df.to_excel(writer, sheet_name="Raw Closing Prices")
    daily_pct_change.to_excel(writer, sheet_name="Daily Pct Change")
    cumulative_movements.to_excel(writer, sheet_name="Cumulative Movements")

# Optional: Preview the structured baseline performance final product
print("Excel Workbook 'global_indices_report.xlsx' created successfully with 3 sheets.")

# --- CORRELATION ANALYSIS ---

# 4e. Correlation matrix using daily percent changes
correlation_matrix = daily_pct_change.corr()

# --- IDENTIFY WEAK & NO CORRELATION PAIRS ---

weak_pairs = []
no_pairs = []

for col1 in correlation_matrix.columns:
    for col2 in correlation_matrix.columns:
        if col1 < col2:  # avoid duplicates & self-pairs
            corr_val = correlation_matrix.loc[col1, col2]
            if abs(corr_val) < 0.10:
                no_pairs.append((col1, col2, corr_val))
            elif abs(corr_val) < 0.30:
                weak_pairs.append((col1, col2, corr_val))

weak_df = pd.DataFrame(weak_pairs, columns=["Asset 1", "Asset 2", "Correlation"])
no_df = pd.DataFrame(no_pairs, columns=["Asset 1", "Asset 2", "Correlation"])

# --- BUILD LOW-CORRELATION PORTFOLIO ---

avg_corr = correlation_matrix.abs().mean().sort_values()

# Select the lowest-correlation assets (top 5)
low_corr_assets = avg_corr.head(5).index.tolist()

portfolio_weights = pd.Series(
    [1/len(low_corr_assets)] * len(low_corr_assets),
    index=low_corr_assets,
    name="Equal Weight"
)

portfolio_df = portfolio_weights.to_frame()


# 4f. Export correlation matrix to Excel
with pd.ExcelWriter("global_indices_report.xlsx", mode="a", engine="openpyxl") as writer:
    correlation_matrix.to_excel(writer, sheet_name="Correlation Matrix")
    weak_df.to_excel(writer, sheet_name="Weak Correlations", index=False)
    no_df.to_excel(writer, sheet_name="No Correlations", index=False)
    portfolio_df.to_excel(writer, sheet_name="Zero-Corr Portfolio")
