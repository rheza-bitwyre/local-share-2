import asyncio
import hashlib
import hmac
import json
import uuid
import websockets

API_KEY = "7rz0k4_00GoEtQW4td6mAM-7jxFCOLoFgKJzVD3x"
SECRET_KEY = "Oj3qyVUEufRp0cYtRMmWg9h5c3A"
CONTROL_URL = "wss://api.bitwyre.com/ws/private/orders/control/v2"
STATUS_URL = "wss://api.bitwyre.com/ws/private/orders/status"

def compute_signature(secret_key):
    return hmac.new(secret_key.encode("utf-8"), b"", hashlib.sha512).hexdigest()

def build_headers():
    api_sign = compute_signature(SECRET_KEY)
    return [
        ("API-Data", json.dumps({
            "api_key": API_KEY,
            "api_sign": api_sign
        }))
    ]

async def get_active_orders():
    async with websockets.client.connect(STATUS_URL, extra_headers=build_headers()) as ws:
        request_data = {
            "command": "get",
            "payload": json.dumps({
                "orderid": "all",
                "instrument": "doge_idr_spot"
            })
        }
        await ws.send(json.dumps(request_data))
        response = await ws.recv()
        parsed = json.loads(response)

        # ✅ Handle list or dict response
        if isinstance(parsed, list):
            for msg in parsed:
                if isinstance(msg, dict) and "orders" in msg:
                    orders = msg["orders"]
                    break
            else:
                orders = []
        else:
            orders = parsed.get("orders", [])

        print("📋 Active Orders:")
        print(json.dumps(orders, indent=2))
        return orders

async def post_order():
    request_id = str(uuid.uuid4())
    payload = {
        "instrument": "doge_idr_spot",
        "ordtype": 2,
        "side": 1,
        "price": "1500",
        "orderqty": "1"
    }
    request_data = {
        "command": "create",
        "payload": json.dumps(payload),
        "request_id": request_id
    }
    async with websockets.client.connect(CONTROL_URL, extra_headers=build_headers()) as ws:
        await ws.send(json.dumps(request_data))
        print(f"✅ Order Create Sent (request_id: {request_id})")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                parsed = json.loads(response)
                print("📥 Create Order Response:")
                print(json.dumps(parsed, indent=2))
        except asyncio.TimeoutError:
            print("⌛ No more messages from control server (timeout).")

async def cancel_order(order_id):
    request_data = {
        "command": "cancel",
        "payload": json.dumps({
            "orderid": order_id,
            "instrument": "doge_idr_spot"
        }),
        "request_id": str(uuid.uuid4())
    }
    async with websockets.client.connect(CONTROL_URL, extra_headers=build_headers()) as ws:
        await ws.send(json.dumps(request_data))
        print(f"❌ Cancel Order Sent (order_id: {order_id})")
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                parsed = json.loads(response)
                print("📥 Cancel Order Response:")
                print(json.dumps(parsed, indent=2))
        except asyncio.TimeoutError:
            print("⌛ No more messages from control server (timeout after cancel).")

async def main():
    print("📋 Step 1 - BEFORE Placing Order Active Orders:")
    await get_active_orders()

    print("\n🛒 Step 2 - Placing Order...")
    await post_order()

    print("\n📋 Step 3 - AFTER Placing Order Active Orders:")
    orders = await get_active_orders()

    if not orders:
        print("⚠️ No active orders found to cancel.")
        return

    order_id = orders[0].get("orderid")
    if order_id:
        print(f"\n❌ Step 4 - Cancelling Order (order_id: {order_id})...")
        await cancel_order(order_id)
    else:
        print("⚠️ Could not find order_id to cancel.")

    print("\n📋 Step 5 - AFTER Cancelling Order Active Orders:")
    await get_active_orders()

    print("\n✅ Step 6 - Done.")

if __name__ == "__main__":
    asyncio.run(main())
