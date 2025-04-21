import asyncio
import hashlib
import hmac
import json
import uuid
import websockets

API_KEY = "7rz0k4_00GoEtQW4td6mAM-7jxFCOLoFgKJzVD3x"
SECRET_KEY = "Oj3qyVUEufRp0cYtRMmWg9h5c3A"
CONTROL_URL = "wss://api.bitwyre.com/ws/private/orders/control/v2"
STREAM_URL = "wss://api.bitwyre.com/ws/private/orders/stream/v2"
STATUS_URL = "wss://api.bitwyre.com/ws/private/orders/status"

def compute_signature(secret_key):
    return hmac.new(secret_key.encode("utf-8"), b"", hashlib.sha512).hexdigest()

async def post_order():
    api_sign = compute_signature(SECRET_KEY)
    headers = {
        "API-Data": json.dumps({
            "api_key": API_KEY,
            "api_sign": api_sign
        })
    }
    payload = {
        "instrument": "doge_idr_spot",
        "ordtype": 2,  # Limit
        "side": 1,     # Sell
        "price": "1500",
        "orderqty": "1"
    }
    request_id = str(uuid.uuid4())
    request_data = {
        "command": "create",
        "payload": json.dumps(payload),
        "request_id": request_id
    }

    async with websockets.client.connect(CONTROL_URL, extra_headers=headers.items()) as ws:
        await ws.send(json.dumps(request_data))
        print("✅ Order create command sent. Waiting for updates...\n")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                parsed = json.loads(response)
                print("📥 Control response:")
                print(json.dumps(parsed, indent=2))
        except asyncio.TimeoutError:
            print("⌛ No more messages from control server (timeout).")

async def poll_order_status(order_id="all", instrument="doge_idr_spot", repeat=3, interval=5):
    api_sign = compute_signature(SECRET_KEY)
    headers = {
        "API-Data": json.dumps({
            "api_key": API_KEY,
            "api_sign": api_sign
        })
    }

    async with websockets.client.connect(STATUS_URL, extra_headers=headers.items()) as ws:
        print("📡 Connected to order status WebSocket.")
        for i in range(repeat):
            request_data = {
                "command": "get",
                "payload": json.dumps({
                    "orderid": order_id,
                    "instrument": instrument
                })
            }
            await ws.send(json.dumps(request_data))
            response = await ws.recv()
            parsed = json.loads(response)
            print(f"🔎 Status check #{i + 1}:")
            print(json.dumps(parsed, indent=2))
            await asyncio.sleep(interval)

async def main():
    await asyncio.gather(
        post_order(),
        poll_order_status(order_id="all", instrument="doge_idr_spot", repeat=20, interval=5)
    )

if __name__ == "__main__":
    asyncio.run(main())
