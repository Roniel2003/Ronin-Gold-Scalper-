"""
MT5 Connection Diagnostic Script
Debug MT5 connection issues step by step.
"""

import MetaTrader5 as mt5
import json
import sys
from datetime import datetime

def test_mt5_connection():
    """Test MT5 connection step by step"""
    
    print("🔍 MT5 Connection Diagnostics")
    print("=" * 50)
    
    # Load config
    try:
        with open('config/mt5_trading.json', 'r') as f:
            config = json.load(f)
        broker_config = config['broker']
        print(f"✅ Config loaded successfully")
        print(f"   Server: {broker_config['server']}")
        print(f"   Login: {broker_config['login']}")
        print(f"   Password: {'*' * len(str(broker_config['password']))}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Step 1: Initialize MT5
    print("\n📡 Step 1: Initialize MT5...")
    path = broker_config.get('path', '')
    
    if path:
        print(f"   Using MT5 path: {path}")
        if not mt5.initialize(path=path):
            print(f"❌ MT5 initialize with path failed")
            print(f"   Error: {mt5.last_error()}")
            # Try without path
            print("   Trying without path...")
            if not mt5.initialize():
                print(f"❌ MT5 initialize without path also failed")
                print(f"   Error: {mt5.last_error()}")
                return False
    else:
        if not mt5.initialize():
            print(f"❌ MT5 initialize failed")
            print(f"   Error: {mt5.last_error()}")
            return False
    
    print("✅ MT5 initialized successfully")
    
    # Step 2: Check MT5 version and info
    print("\n📋 Step 2: MT5 Information...")
    try:
        version = mt5.version()
        if version:
            print(f"   MT5 Version: {version}")
        else:
            print("   Could not get MT5 version")
    except Exception as e:
        print(f"   Error getting version: {e}")
    
    # Step 3: Login attempt
    print("\n🔐 Step 3: Login to account...")
    server = broker_config['server']
    login = int(broker_config['login'])
    password = str(broker_config['password'])
    
    print(f"   Attempting login to {server}...")
    
    if not mt5.login(login, password=password, server=server):
        error = mt5.last_error()
        print(f"❌ Login failed")
        print(f"   Error Code: {error[0]}")
        print(f"   Error Message: {error[1]}")
        
        # Common error codes and solutions
        error_code = error[0]
        if error_code == 10004:
            print("💡 Solution: Invalid account credentials")
        elif error_code == 10015:
            print("💡 Solution: Invalid server name")
        elif error_code == 10016:
            print("💡 Solution: Old client version")
        elif error_code == 10017:
            print("💡 Solution: No connection to server")
        elif error_code == 10018:
            print("💡 Solution: Account disabled")
        else:
            print(f"💡 Check MT5 documentation for error code {error_code}")
        
        mt5.shutdown()
        return False
    
    print("✅ Login successful")
    
    # Step 4: Get account info
    print("\n💰 Step 4: Account Information...")
    account_info = mt5.account_info()
    if account_info:
        print(f"   Account: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Equity: ${account_info.equity:,.2f}")
        print(f"   Currency: {account_info.currency}")
        print(f"   Company: {account_info.company}")
    else:
        print("❌ Could not get account info")
        print(f"   Error: {mt5.last_error()}")
    
    # Step 5: Test symbol access
    print("\n📈 Step 5: Symbol Access...")
    test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    
    for symbol in test_symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            print(f"   ✅ {symbol}: Available (Spread: {symbol_info.spread})")
            
            # Try to get tick data
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"      Current price: {tick.bid}/{tick.ask}")
            else:
                print(f"      Could not get tick data")
        else:
            print(f"   ❌ {symbol}: Not available")
            # Try to select symbol
            if mt5.symbol_select(symbol, True):
                print(f"      Symbol selected successfully")
            else:
                print(f"      Could not select symbol")
    
    # Cleanup
    mt5.shutdown()
    print("\n✅ Diagnostic complete")
    return True

if __name__ == "__main__":
    try:
        test_mt5_connection()
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
