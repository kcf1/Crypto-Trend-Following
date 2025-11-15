from typing import Final

# ------------------------------------------------------------------ #
# Table: klines (interval = 1h only → no column)
# ------------------------------------------------------------------ #
CREATE_KLINES_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS klines (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,        -- ms from Binance
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time INTEGER NOT NULL,
    quote_volume REAL NOT NULL,
    num_trades INTEGER NOT NULL,
    taker_buy_base_vol REAL NOT NULL,
    taker_buy_quote_vol REAL NOT NULL,
    PRIMARY KEY (symbol, open_time)
);
"""
INDEX_KLINES: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_klines_symbol_time 
ON klines (symbol, open_time DESC);
"""

CREATE_TRADES_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    quote_qty REAL NOT NULL,
    time INTEGER NOT NULL,
    is_buyer_maker INTEGER NOT NULL,
    is_best_match INTEGER NOT NULL,
    UNIQUE(symbol, id)
);
"""
INDEX_TRADES: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
ON trades (symbol, time DESC);
"""

CREATE_SYMBOLS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT NOT NULL,
    symbol_bnb TEXT NOT NULL,
    class TEXT NOT NULL,
    currency_base TEXT NOT NULL,
    currency_profit TEXT NOT NULL,
    trade_contract_value REAL NOT NULL,
    trade_contract_size REAL NOT NULL,
    trade_tick_value REAL NOT NULL,
    trade_tick_size REAL NOT NULL,
    PRIMARY KEY (symbol)
);
"""
INDEX_SYMBOLS: Final[str] = """
"""

CREATE_MTBARS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS mtbars (
    symbol TEXT NOT NULL,
    time INTEGER NOT NULL,     -- ms (UTC)
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread INTEGER NOT NULL,
    real_volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, time)
);
"""
INDEX_MTBARS: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_mtbars_symbol_time 
ON mtbars (symbol, time DESC);
"""