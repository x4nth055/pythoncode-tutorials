from fxcmpy import fxcmpy

# generate this once you create your demo account
# this is fake token, just for demonstration
ACCESS_TOKEN = "8438834e8edaff70ca3db0088a8d6c5c37f51279"

try:
    fxcm_con = fxcmpy(access_token=ACCESS_TOKEN, server="demo")
    print("Is connected:", fxcm_con.is_connected())

except Exception as e:
    print(e)


fxcm_con.open_trade(symbol="US30",amount=1,is_buy=True,time_in_force="GTC",order_type="AtMarket")

trade_id = fxcm_con.get_open_trade_ids()[0]
print("Closing trade:", trade_id)
fxcm_con.close_trade(trade_id=trade_id,amount=1)

fxcm_con.open_trade(symbol="US30",amount=1,is_buy=True,time_in_force="GTC",order_type="AtMarket",is_in_pips=True,limit=15,stop=-50)

fxcm_con.close()