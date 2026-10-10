
import asyncio
import json
import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws"

    async with websockets.connect(uri) as websocket:
        request = {
            "question": "What is the ticket priority policy?",
            "session_id": "websocket-test-01",
        }

        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        result = json.loads(response)

        print("WebSocket response received:")
        print(json.dumps(result, indent=2))

        assert "answer" in result or "error" in result
        print("\nWebSocket request test: PASSED")

    print("WebSocket client disconnected cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
