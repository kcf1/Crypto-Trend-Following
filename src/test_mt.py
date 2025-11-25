import pandas as pd
from api_bnb import get_klines
from api_mt5 import get_mt5_symbols,init_mt5,get_mt5_bars
from db_load import init_db,save_symbols
from db_read import read_klines,read_mtbars
from datetime import datetime

init_mt5()

start_time,end_time = datetime(2025,11,15),datetime.now()
df = get_klines('BTCUSDT',start_time=start_time,end_time=end_time).tail()
df['open_time'] = pd.to_datetime(df['open_time'],unit='ms')
print(df)
df = read_klines('BTCUSDT',start_time=start_time,end_time=end_time).tail()
df['open_time'] = pd.to_datetime(df['open_time'],unit='ms')
print(df)
df = get_mt5_bars('BTCUSD',date_from=start_time,date_to=end_time).tail()
df['time'] = pd.to_datetime(df['time'],unit='ms')
print(df)
df = read_mtbars('BTCUSD',start_time=start_time,end_time=end_time).tail()
print(df)