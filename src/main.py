from config import TZ,MODEL_DIR,logger
from api_mt5 import init_mt5,shutdown_mt5
from port_rebalance import rebalance_portfolio
from data_update import update_all_data
from utils import onhour_offset
from datetime import datetime

logger.info("Starting rebalancing session...")
init_mt5()

onhour_offset(offset_mins=0,offset_secs=-15)
update_all_data()

# run at 05 if 4am market close, at 00 if not
if datetime.now(TZ).hour == 22: onhour_offset(offset_mins=-5,offset_secs=-15)
rebalance_portfolio()

shutdown_mt5()

logger.success("Done!")