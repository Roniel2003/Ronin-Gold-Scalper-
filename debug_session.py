#!/usr/bin/env python3
"""
Debug script to test session detection for gold market
"""

import pandas as pd
import pytz
from datetime import datetime, time
import json
from ronin_engine_v3 import is_in_session_v3

def load_config():
    """Load MT5 trading configuration"""
    try:
        with open('config/mt5_trading.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Config file not found, using defaults")
        return {}

def debug_session_detection():
    """Debug current session detection logic"""
    
    # Load configuration
    config = load_config()
    print("=== GOLD MARKET SESSION DEBUG ===")
    print(f"Current time: {datetime.now()}")
    
    # Get current time in different timezones
    utc_now = pd.Timestamp.now(tz=pytz.UTC)
    est_now = utc_now.tz_convert('US/Eastern')
    
    print(f"UTC time: {utc_now}")
    print(f"EST time: {est_now}")
    print(f"EST time only: {est_now.time()}")
    
    # Check session configuration
    print("\n=== SESSION CONFIGURATION ===")
    sessions_config = config.get('sessions', {})
    
    # Session 1 (Asian/Default)
    session_start_hour = sessions_config.get('session_start_hour', 19)
    session_start_minute = sessions_config.get('session_start_minute', 0)
    session_end_hour = sessions_config.get('session_end_hour', 1)
    session_end_minute = sessions_config.get('session_end_minute', 0)
    
    print(f"Session 1: {session_start_hour:02d}:{session_start_minute:02d} - {session_end_hour:02d}:{session_end_minute:02d} EST")
    
    # Session 2 (London/NY Overlap)
    session_start_hour_2 = sessions_config.get('session_start_hour_2', 8)
    session_start_minute_2 = sessions_config.get('session_start_minute_2', 0)
    session_end_hour_2 = sessions_config.get('session_end_hour_2', 12)
    session_end_minute_2 = sessions_config.get('session_end_minute_2', 0)
    
    print(f"Session 2: {session_start_hour_2:02d}:{session_start_minute_2:02d} - {session_end_hour_2:02d}:{session_end_minute_2:02d} EST")
    
    # Test session detection
    print("\n=== SESSION DETECTION TEST ===")
    
    # Test with current time
    is_in_session_now = is_in_session_v3(est_now, config)
    print(f"Current time in session: {is_in_session_now}")
    
    # Test manual session logic
    current_time = est_now.time()
    
    # Session 1 logic (crosses midnight check)
    session_start = time(session_start_hour, session_start_minute)
    session_end = time(session_end_hour, session_end_minute)
    
    if session_start > session_end:  # Crosses midnight
        in_session_1 = current_time >= session_start or current_time <= session_end
    else:
        in_session_1 = session_start <= current_time <= session_end
    
    # Session 2 logic
    session_start_2 = time(session_start_hour_2, session_start_minute_2)
    session_end_2 = time(session_end_hour_2, session_end_minute_2)
    in_session_2 = session_start_2 <= current_time <= session_end_2
    
    print(f"Session 1 active: {in_session_1}")
    print(f"Session 2 active: {in_session_2}")
    print(f"Overall in session: {in_session_1 or in_session_2}")
    
    # Gold market hours info
    print("\n=== GOLD MARKET HOURS INFO ===")
    print("Gold market is typically open:")
    print("Sunday 5:00 PM EST - Friday 5:00 PM EST")
    print("24 hours during weekdays")
    
    # Check if it's weekend
    weekday = est_now.weekday()  # 0=Monday, 6=Sunday
    print(f"Current weekday: {weekday} (0=Mon, 6=Sun)")
    
    if weekday == 5:  # Saturday
        print("WARNING: It's Saturday - Gold market is CLOSED")
    elif weekday == 6:  # Sunday
        if est_now.hour < 17:
            print("WARNING: It's Sunday before 5 PM EST - Gold market is CLOSED")
        else:
            print("It's Sunday after 5 PM EST - Gold market should be OPEN")
    else:  # Monday-Friday
        print("It's a weekday - Gold market should be OPEN (unless major holiday)")

if __name__ == "__main__":
    debug_session_detection()
