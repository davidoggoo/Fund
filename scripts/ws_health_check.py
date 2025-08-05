#!/usr/bin/env python3
"""
WebSocket Health Check Script for CI
Connects to RTAI WebSocket and verifies it receives data within timeout
"""

import asyncio
import websockets
import json
import sys
import time
from typing import List

async def websocket_health_check(
    url: str = "ws://localhost:8000/ws",
    timeout: int = 10,
    min_messages: int = 3
) -> bool:
    """
    Check WebSocket health by connecting and receiving messages
    
    Args:
        url: WebSocket URL to test
        timeout: Maximum time to wait for messages
        min_messages: Minimum number of messages to receive
        
    Returns:
        bool: True if health check passes
    """
    print(f"🔍 WebSocket Health Check: {url}")
    print(f"   Timeout: {timeout}s, Min messages: {min_messages}")
    
    received_messages: List[dict] = []
    start_time = time.time()
    
    try:
        async with websockets.connect(url) as websocket:
            print("✅ WebSocket connected successfully")
            
            while len(received_messages) < min_messages:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"❌ Timeout after {elapsed:.1f}s")
                    print(f"   Only received {len(received_messages)}/{min_messages} messages")
                    return False
                
                try:
                    # Wait for message with short timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    
                    try:
                        data = json.loads(message)
                        received_messages.append(data)
                        
                        # Log message types for debugging
                        message_type = data.get('type', 'unknown')
                        print(f"📨 Received message #{len(received_messages)}: {message_type}")
                        
                        if 'symbol' in data:
                            print(f"   Symbol: {data['symbol']}")
                        if 'price' in data:
                            print(f"   Price: {data['price']}")
                            
                    except json.JSONDecodeError:
                        print(f"⚠️  Received non-JSON message (skipping): {message[:100]}...")
                        continue
                        
                except asyncio.TimeoutError:
                    # No message in 2s, continue waiting
                    continue
                    
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - is the server running?")
        return False
    except websockets.exceptions.InvalidURI:
        print(f"❌ Invalid WebSocket URI: {url}")
        return False
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return False
    
    elapsed = time.time() - start_time
    print(f"✅ Health check passed in {elapsed:.1f}s")
    print(f"   Received {len(received_messages)} messages")
    
    # Analyze message types
    message_types = {}
    for msg in received_messages:
        msg_type = msg.get('type', 'unknown')
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    print("📊 Message type summary:")
    for msg_type, count in message_types.items():
        print(f"   {msg_type}: {count}")
    
    return True

def main():
    """Main health check function"""
    try:
        # Run health check
        result = asyncio.run(websocket_health_check())
        
        if result:
            print("\n🎉 WebSocket health check PASSED")
            sys.exit(0)
        else:
            print("\n💥 WebSocket health check FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Health check error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
