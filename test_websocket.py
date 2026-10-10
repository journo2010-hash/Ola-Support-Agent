
import asyncio
import json
import websockets


async def main():
    # Connect to the API server running on port 8001.
    uri = "ws://127.0.0.1:8001/ws"

    request = {
        "question": "What is the ticket priority policy?",
        "session_id": "websocket-test-01",
    }

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(request))

            response = await websocket.recv()
            result = json.loads(response)

            print("WebSocket response received:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # Verify the expected response fields.
            assert result.get("answer"), "Response answer is missing."
            assert result.get("mode") == "MOCK_LLM", (
                "Unexpected response mode."
            )
            assert result.get("fallback") is False, (
                "The policy question unexpectedly used fallback."
            )
            assert "ticket_priority.md" in result.get("sources", []), (
                "Expected policy source was not returned."
            )
            assert result.get("governance", {}).get("allowed") is True, (
                "The read-only policy request was not allowed."
            )

            print("\nWebSocket request test: PASSED")

    except AssertionError:
        raise
    except Exception as exc:
        print(
            f"WebSocket request test failed: "
            f"{type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())
