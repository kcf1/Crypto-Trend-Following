# mt5_init.py
import MetaTrader5 as mt5
import time
from typing import Optional, Final, Dict, Any, Union, List
from config import MT5_SERVER, MT5_LOGIN, MT5_PASSWORD, MT5_TZ, TZ, logger
from threading import Lock
import atexit
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from decimal import Decimal, ROUND_HALF_UP
import time

# Mapping: string interval → MT5 TIMEFRAME constant
_INTERVAL_TO_TF: Dict[str, int] = {
    # Minutes
    '1m':  mt5.TIMEFRAME_M1,
    '2m':  mt5.TIMEFRAME_M2,
    '3m':  mt5.TIMEFRAME_M3,
    '4m':  mt5.TIMEFRAME_M4,
    '5m':  mt5.TIMEFRAME_M5,
    '6m':  mt5.TIMEFRAME_M6,
    '10m': mt5.TIMEFRAME_M10,
    '12m': mt5.TIMEFRAME_M12,
    '15m': mt5.TIMEFRAME_M15,
    '20m': mt5.TIMEFRAME_M20,
    '30m': mt5.TIMEFRAME_M30,

    # Hours
    '1h':  mt5.TIMEFRAME_H1,
    '2h':  mt5.TIMEFRAME_H2,
    '3h':  mt5.TIMEFRAME_H3,
    '4h':  mt5.TIMEFRAME_H4,
    '6h':  mt5.TIMEFRAME_H6,
    '8h':  mt5.TIMEFRAME_H8,
    '12h': mt5.TIMEFRAME_H12,

    # Days, Weeks, Months
    '1d':  mt5.TIMEFRAME_D1,
    '1w':  mt5.TIMEFRAME_W1,
    '1M':  mt5.TIMEFRAME_MN1,  # MN1 = month
}

# ------------------------------------------------------------------ #
# Thread-safe lock (optional, for multi-threaded use)
# ------------------------------------------------------------------ #
_mt5_lock = Lock()
_mt5_initialized = False

# ------------------------------------------------------------------ #
# 1. Initialize MT5
# ------------------------------------------------------------------ #
def init_mt5(
    server: Optional[str] = None,
    login: Optional[int] = None,
    password: Optional[str] = None,
    timeout: int = 10000,
    retries: int = 3,
    delay: float = 2.0
) -> bool:
    """
    Initialize MT5 connection using config or overrides.

    Args:
        server: MT5 server (e.g. 'FTMO-Demo')
        login: Account login
        password: Account password
        timeout: Connection timeout in ms
        retries: Number of retry attempts
        delay: Delay between retries

    Returns:
        bool: True if connected and logged in
    """
    global _mt5_initialized

    server = server or MT5_SERVER
    login = login or MT5_LOGIN
    password = password or MT5_PASSWORD
    print(server,login,password)

    if not all([server, login, password]):
        logger.error("MT5 credentials missing in .env or args")
        return False

    with _mt5_lock:
        if _mt5_initialized:
            logger.debug("MT5 already initialized")
            return True

        logger.info(f"Connecting to MT5: {server} | Login: {login}")

        for attempt in range(1, retries + 1):
            if mt5.initialize(
                server=server,
                login=login,
                password=password,
                timeout=timeout
            ):
                # Check login
                account_info = mt5.account_info()
                if account_info and account_info.login == login:
                    _mt5_initialized = True
                    logger.success(f"MT5 connected | Balance: {account_info.balance} {account_info.currency}")
                    atexit.register(shutdown_mt5)  # Auto-shutdown on exit
                    return True
                else:
                    logger.warning(f"MT5 initialized but login failed (attempt {attempt})")
            else:
                logger.warning(f"MT5 init failed: {mt5.last_error()} (attempt {attempt})")

            if attempt < retries:
                time.sleep(delay)

        logger.error("MT5 initialization failed after all retries")
        return False

def read_balance() -> Dict[str, Any]:
    """
    Read current MT5 account balance and stats.

    Returns:
        dict with:
            balance, equity, margin, free_margin,
            margin_level, profit, credit, currency
    """
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        logger.error(f"Failed to read account: {mt5.last_error()}")
        raise RuntimeError("MT5 account info not available")

    # Extract key fields
    stats = {
        'balance': account_info.balance,
        'equity': account_info.equity,
        'margin': account_info.margin,
        'free_margin': account_info.margin_free,
        'margin_level': account_info.margin_level,
        'profit': account_info.profit,
        'credit': account_info.credit,
        'currency': account_info.currency,
        'leverage': account_info.leverage,
        'margin_mode': account_info.margin_mode,
        'trade_allowed': account_info.trade_allowed,
        'trade_expert': account_info.trade_expert
    }

    logger.info(
        f"Account: {stats['balance']:.2f} {stats['currency']} | "
        f"Equity: {stats['equity']:.2f} | "
        f"Margin: {stats['margin']:.2f} | "
        f"Free: {stats['free_margin']:.2f} | "
        f"Level: {stats['margin_level']:.1f}%"
    )

    return stats

# ------------------------------------------------------------------ #
# 2. Shutdown MT5
# ------------------------------------------------------------------ #
def shutdown_mt5() -> None:
    """Safely shutdown MT5 connection."""
    global _mt5_initialized
    with _mt5_lock:
        if _mt5_initialized:
            mt5.shutdown()
            _mt5_initialized = False
            logger.info("MT5 connection closed")
        else:
            logger.debug("MT5 not initialized — nothing to shutdown")

# ------------------------------------------------------------------ #
# 3. Health check
# ------------------------------------------------------------------ #
def is_mt5_ready() -> bool:
    """Check if MT5 is connected and responsive."""
    if not _mt5_initialized:
        return False
    # Try a lightweight call
    return mt5.terminal_info() is not None

def get_mt5_symbols(
    asset_class: str = None,
) -> pd.DataFrame:
    """
    Fetch MT5 symbols and return a clean DataFrame with only required columns.
    
    Returns:
        pd.DataFrame with columns:
        - symbol
        - class
        - currency_base
        - currency_profit
        - price
        - spread
        - trade_contract_value
        - trade_contract_size
        - trade_tick_value
        - trade_tick_size
    """
    # Validate inputs
    asset_classes = [
        'Crypto', 'Metals', 'Commodities', 'Agriculture',
        'Forex', 'Exotics', 'Cash', 'Equities'
    ]
    if asset_class and asset_class not in asset_classes:
        raise ValueError(f"Invalid asset class. Choose from: {', '.join(asset_classes)}")
    
    if not is_mt5_ready():
        logger.error("MT5 initialization failed")
        raise RuntimeError("Failed to initialize MT5")

    logger.info("Fetching symbols from MT5...")
    raw_symbols = mt5.symbols_get()
    
    if not raw_symbols:
        logger.warning("No symbols returned from MT5")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([s._asdict() for s in raw_symbols])
    
    # Extract 'class' from path (e.g., "Crypto II CFD\XTZUSD" → "Crypto II CFD")
    df['class'] = df['path'].str.replace('\\', ' ').str.split().str[0]
    if asset_class:
        df = df[df['class']==asset_class]

    df['price'] = (df['bid'] + df['ask'])/2
    df['spread'] = df['ask'] - df['bid']
    
    # Compute trade_contract_value
    df['trade_contract_value'] = df['trade_tick_value'] / df['trade_tick_size'] * df['price']
    
    # Select only needed columns
    selected_cols = [
        'name',                    # → symbol
        'class',
        'currency_base',
        'currency_profit',
        'price',
        'spread',
        'trade_contract_value',
        'trade_contract_size',
        'trade_tick_value',
        'trade_tick_size'
    ]
    df = df[selected_cols].copy()
    
    # Rename to match DB
    df = df.rename(columns={'name': 'symbol'})
    df['symbol_bnb'] = df['symbol']+'T'
    
    # Ensure types
    df['price'] = df['price'].round(8)
    df['spread'] = df['spread'].round(8)
    df['trade_contract_value'] = df['trade_contract_value'].round(8)
    df['trade_tick_value'] = df['trade_tick_value'].round(8)
    df['trade_tick_size'] = df['trade_tick_size'].round(8)
    
    logger.success(f"Fetched {len(df)} symbols from MT5")
    return df

def get_mt5_bars(
    symbol: str,
    date_from: datetime,
    date_to: datetime,
    interval: str = '1h',
) -> pd.DataFrame:
    """
    Download 1H bars from MT5 (time in GMT+2) → convert to UTC.

    Args:
        symbol: e.g. "EURUSD"
        date_from: UTC datetime (inclusive)
        date_to: UTC datetime (inclusive)

    Returns:
        pd.DataFrame with UTC index
    """
    # Validate inputs
    valid_intervals = [
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '1w', '1M'
    ]
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Choose from: {', '.join(valid_intervals)}")

    # Convert input to MT5's timezone (GMT+2) for correct range
    date_from_mt5 = date_from.astimezone(MT5_TZ).replace(tzinfo=pytz.UTC) if date_from.tzinfo else MT5_TZ.localize(date_from).replace(tzinfo=pytz.UTC)
    date_to_mt5 = date_to.astimezone(MT5_TZ).replace(tzinfo=pytz.UTC) if date_to.tzinfo else MT5_TZ.localize(date_to).replace(tzinfo=pytz.UTC)
    timeframe_mt5 = _INTERVAL_TO_TF[interval]

    logger.info(f"Downloading {symbol} {interval} bars: {date_from} → {date_to} (UTC)")

    rates = mt5.copy_rates_range(symbol, timeframe_mt5, date_from_mt5, date_to_mt5)
    if rates is None or len(rates) == 0:
        logger.warning(f"No {interval} bars for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    # MT5 gives time in GMT+2 → localize and convert to UTC
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=False)
    df['time'] = df['time'].dt.tz_localize(MT5_TZ).dt.tz_convert(TZ)
    df['time'] = df['time'].astype(int) // 1_000_000

    # Set index
    df = df.sort_values('time')
    #df = df.set_index('time').sort_index()

    logger.success(f"Downloaded {len(df)} {interval} bars (converted to UTC)")
    return df

# ------------------------------------------------------------------ #
# 1. Send new market order
# ------------------------------------------------------------------ #
def order_send(
    symbol: str,
    quantity: float,
    ticket: Optional[int] = None,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
    comment: Optional[str] = None,
    magic: int = 123456
) -> Dict[str, Any]:
    """
    Send a market order (buy/sell) or modify existing position.

    Args:
        symbol: str
        quantity: float (positive = buy, negative = sell) — **order volume**
        ticket: position ticket (optional)
        tp: take profit price
        sl: stop loss price
        comment: optional comment
        magic: EA magic number

    Returns:
        dict: order result
    """
    if quantity == 0:
        raise ValueError("Quantity cannot be zero")

    # Round volume to 2 decimals
    volume_decimal = Decimal(str(abs(quantity))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    volume = float(volume_decimal)

    # Determine order direction
    order_type = mt5.ORDER_TYPE_BUY if quantity > 0 else mt5.ORDER_TYPE_SELL

    # --- Case: Modify existing position ---
    if ticket is not None:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise ValueError(f"Position not found: ticket={ticket}")

        pos = positions[0]
        if pos.symbol != symbol:
            raise ValueError(f"Symbol mismatch: {pos.symbol} vs {symbol}")

        current_volume = pos.volume
        current_type = pos.type  # 0=BUY, 1=SELL
        current_sign = 1 if current_type == mt5.ORDER_TYPE_BUY else -1
        order_sign = 1 if quantity > 0 else -1

        # Rule 1: Same direction → cannot increase
        if current_sign == order_sign:
            raise ValueError(
                f"Cannot increase position. Current: {current_volume}, Order: {volume}"
            )
        # Rule 2: Different direction → cannot reverse
        else:
            if volume > current_volume:
                raise ValueError(
                    f"Cannot reverse position. Current: {'BUY' if current_sign > 0 else 'SELL'}, "
                    f"Order: {'BUY' if order_sign > 0 else 'SELL'}"
                )
            # Allow reduce → partial close
            if volume <= current_volume:
                # Close difference with opposite order
                reduce_volume = volume
                close_type = mt5.ORDER_TYPE_SELL if current_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                logger.info(f"Reducing position: {current_volume} → {current_volume-volume} (close {volume})")

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": reduce_volume,
                    "type": close_type,
                    "position": ticket,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "comment": comment or "partial_close"
                }

    # --- Case: New order ---
    else:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
            "magic": magic,
            "comment": comment or ""
        }

        if tp is not None:
            tp_decimal = Decimal(str(tp)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            request["tp"] = float(tp_decimal)
        if sl is not None:
            sl_decimal = Decimal(str(sl)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            request["sl"] = float(sl_decimal)

    # --- Send order ---
    logger.info(f"Sending order: {symbol}, {quantity:+.2f} lots, TP={tp}, SL={sl}, ticket={ticket}")

    result = mt5.order_send(request)
    result_dict = result._asdict()
    result_dict['request'] = request

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        time.sleep(0.01)
        try:
            order = mt5.history_orders_get(ticket=result.order)[0]
            result_dict['price'] = order.price_current
        except:
            result_dict['price'] = None
        logger.success(f"Order executed: {symbol} {quantity:+.2f} @ {result_dict['price']}")
    else:
        logger.error(f"Order failed: {result.retcode} - {result.comment}")

    return result_dict

# ------------------------------------------------------------------ #
# 3. Close position by ticket
# ------------------------------------------------------------------ #
def order_close(
    ticket: int,
    comment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Close position by ticket (market close).

    Args:
        ticket: position ticket
        comment: optional comment

    Returns:
        dict: close result
    """
    # Get position
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.error(f"Position not found: ticket={ticket}")
        return {"retcode": -1, "comment": "Position not found"}

    pos = positions[0]
    symbol = pos.symbol
    volume = pos.volume  # already clean (from order_send)

    # Opposite direction
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    quantity = volume if order_type == mt5.ORDER_TYPE_BUY else -volume

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "position": ticket,
        "type_filling": mt5.ORDER_FILLING_FOK,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": comment or "close"
    }

    logger.info(f"Closing position: ticket={ticket}, volume={volume}")

    result = mt5.order_send(request)
    result_dict = result._asdict()
    result_dict['request'] = request

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        time.sleep(0.01)
        order = read_history_orders(ticket=result.order)
        result_dict['price'] = order['price_current']
        logger.success(f"Position closed: ticket={ticket}, {symbol} {quantity:+.2f} @ {result_dict['price']}")
    else:
        logger.error(f"Close failed: {result.retcode} - {result.comment}")

    return result_dict

def read_history_orders(
    date_from: Optional[Union[datetime, int]] = None,
    date_to: Optional[Union[datetime, int]] = None,
    group: Optional[str] = None,
    ticket: Optional[int] = None,
    position: Optional[int] = None
) -> pd.DataFrame:
    """
    Read historical orders from MT5 with correct timezone handling.

    Args:
        date_from: UTC datetime or Unix seconds
        date_to: UTC datetime or Unix seconds
        group: symbol filter
        ticket: single ticket
        position: position ID

    Returns:
        pd.DataFrame with UTC datetime index
    """
    # --- Validate mode ---
    if ticket is not None and any([date_from, date_to, group, position]):
        raise ValueError("Use 'ticket' alone")
    if position is not None and any([date_from, date_to, group, ticket]):
        raise ValueError("Use 'position' alone")

    # --- Convert to datetime if needed ---
    if isinstance(date_from, int):
        date_from = datetime.fromtimestamp(date_from, tz=TZ)
    if isinstance(date_to, int):
        date_to = datetime.fromtimestamp(date_to, tz=TZ)

    # --- Ensure UTC input ---
    if date_from and date_from.tzinfo is None:
        date_from = TZ.localize(date_from)
    if date_to and date_to.tzinfo is None:
        date_to = TZ.localize(date_to)

    # --- Call MT5: convert input to GMT+2 ---
    if ticket is not None:
        logger.info(f"Fetching order by ticket: {ticket}")
        orders = mt5.history_orders_get(ticket=ticket)[0]
        return orders._asdict()
    elif position is not None:
        logger.info(f"Fetching orders by position: {position}")
        orders = mt5.history_orders_get(position=position)[0]
        return orders._asdict()
    else:
        if not date_from or not date_to:
            raise ValueError("date_from and date_to required")
        
        # YOUR METHOD: Convert to MT5 timezone (GMT+2)
        date_from_mt5 = date_from.astimezone(MT5_TZ).replace(tzinfo=pytz.UTC) if date_from.tzinfo else MT5_TZ.localize(date_from).replace(tzinfo=pytz.UTC)
        date_to_mt5 = date_to.astimezone(MT5_TZ).replace(tzinfo=pytz.UTC) if date_to.tzinfo else MT5_TZ.localize(date_to).replace(tzinfo=pytz.UTC)

        logger.info(f"Fetching orders: {date_from} → {date_to} (UTC) | group='{group or 'all'}'")
        orders = mt5.history_orders_get(date_from_mt5, date_to_mt5, group=group)

    # --- Handle result ---
    if orders is None or len(orders) == 0:
        logger.warning("No historical orders found")
        return pd.DataFrame()

    df = pd.DataFrame([o._asdict() for o in orders])

    # --- YOUR METHOD: Convert MT5 time (GMT+2) → UTC ---
    time_cols = ['time_setup', 'time_done']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s', utc=False)
            df[col] = df[col].dt.tz_localize(MT5_TZ).dt.tz_convert(TZ)

    # Set index
    df = df.set_index('time_done').sort_index()

    logger.success(f"Read {len(df)} historical orders (UTC)")
    return df

def read_positions(
    symbol: Optional[str] = None,
    group: Optional[str] = None
) -> pd.DataFrame:
    """
    Read current open positions from MT5.

    Args:
        symbol: filter by symbol (e.g. "EURUSD")
        group: filter by symbol group (e.g. "*USD*", "!*JPY*")

    Returns:
        pd.DataFrame with UTC datetime columns
    """
    # --- Fetch positions ---
    if symbol:
        logger.info(f"Fetching position: {symbol}")
        positions = mt5.positions_get(symbol=symbol)
    elif group:
        logger.info(f"Fetching positions: group='{group}'")
        positions = mt5.positions_get(group=group)
    else:
        logger.info("Fetching all open positions")
        positions = mt5.positions_get()

    # --- Handle result ---
    if positions is None or len(positions) == 0:
        logger.warning(f"No open positions found ({symbol or group or 'all'})")
        return pd.DataFrame()

    df = pd.DataFrame([p._asdict() for p in positions])

    # --- Convert time fields to UTC datetime ---
    time_cols = ['time', 'time_update', 'time_setup']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)

    # --- Clean up ---
    df = df.drop(columns=['external_id'], errors='ignore')  # optional
    df = df.sort_values('time', ascending=False)

    logger.success(f"Read {len(df)} open position(s)")
    return df

def adjust_position(
    symbol: str,
    target_volume: float,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
    comment: Optional[str] = None
) -> List[Dict]:
    """
    Adjust position to target volume using your rules.

    Args:
        symbol: str
        target_volume: float (positive = long, negative = short)
        tp: take profit
        sl: stop loss
        comment: optional

    Returns:
        list of order results
    """
    #if target_volume == 0:
    #    raise ValueError("target_volume=0 not allowed. Use order_close()")

    results = []

    # --- 1. Get current positions ---
    df = read_positions(symbol=symbol)
    if df.empty:
        current_total = 0.0
        current_positions = []
    else:
        # Sum signed volume
        df['signed_volume'] = df['volume'] * np.where(df['type'] == mt5.ORDER_TYPE_BUY, 1, -1)
        current_total = df['signed_volume'].sum()
        current_positions = df.to_dict('records')

    logger.info(f"Current {symbol}: {current_total:+.2f} lots")

    target_sign = 1 if target_volume > 0 else -1
    current_sign = 1 if current_total > 0 else -1 if current_total < 0 else 0
    
    # --- No change ---
    if abs(target_volume - current_total) < 9e-3:
        logger.info("No change needed")
        return []

    # --- SCENE 0: Remain no position ---
    if current_total == 0.0 and target_volume == 0:
        logger.info(f"SCENE 0: Remain no position")
        res = dict()
        results.append(res)
        return results

    # --- SCENE 1: New position ---
    if current_total == 0.0 and current_positions == []:
        logger.info(f"SCENE 1: New order → open {target_volume:+.2f}")
        res = order_send(symbol, target_volume, tp=tp, sl=sl, comment=comment or "new_order")
        results.append(res)
        return results
    
    # --- SCENE 2: Close all (e.g. +0.1 → -0.1) ---
    if target_volume == 0:
        logger.info("SCENE 2: Close all → close {current_total:+.2f}")
        # Close all
        for pos in current_positions:
            res = order_close(pos['ticket'], comment=comment or "close_all")
            results.append(res)
        results.append(res)
        return results

    # --- SCENE 3: Increase (same direction, |target| > |current|) ---
    if current_sign == target_sign and abs(target_volume) > abs(current_total):
        diff = target_volume - current_total
        logger.info(f"SCENE 3: Increase → open {diff:+.2f}")
        res = order_send(symbol, diff, tp=tp, sl=sl, comment=comment or "increase")
        results.append(res)
        return results

    # --- SCENE 4: Reduce (same direction, |target| < |current|) ---
    if current_sign == target_sign and abs(target_volume) < abs(current_total):
        logger.info("SCENE 4: Reduce → close smallest first")
        # Sort by volume ascending
        sorted_pos = sorted(
            current_positions,
            key=lambda x: x['volume'] * (1 if x['type'] == mt5.ORDER_TYPE_BUY else -1)
        )
        if current_sign == -1: 
            sorted_pos = sorted_pos[::-1]

        remaining = current_total - target_volume
        for pos in sorted_pos:
            pos_vol = pos['volume'] * (1 if pos['type'] == mt5.ORDER_TYPE_BUY else -1)
            if remaining == 0:
                break
            if pos_vol == 0:
                continue
            close_vol = min(abs(pos_vol), abs(remaining))
            close_sign = 1 if remaining > 0 else -1
            close_quantity = close_sign * close_vol
            res = order_send(symbol, -close_quantity, ticket=pos['ticket'], comment=comment or "reduce")
            results.append(res)
            remaining -= close_quantity
        return results

    # --- SCENE 5: Reverse direction (e.g. +0.1 → -0.1) ---
    if current_sign != 0 and target_sign != current_sign:
        logger.info("SCENE 5: Reverse → close all, open new")
        # Close all
        for pos in current_positions:
            res = order_close(pos['ticket'], comment=comment or "close_all")
            results.append(res)
        # Open new
        res = order_send(symbol, target_volume, tp=tp, sl=sl, comment=comment or "new_reverse")
        results.append(res)
        return results

    raise RuntimeError("Unhandled case")