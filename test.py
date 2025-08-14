#!/usr/bin/env python3
"""Simple test for vLLM inference server"""
import base64
import time
import asyncio
import json
import websockets
from openai import OpenAI


def test_http_api():
    # Initialize client
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # Read and encode image
    with open("/home/ubuntu/experiment/test-img.png", "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    # Test vision chat
    stream = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What food do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=1024,
        stream=True,
    )

    response = ""
    start_time = time.time()
    first_token = False
    for chunk in stream:
        if chunk.choices[0].delta.content:
            if not first_token:
                first_token = True
                ttft = time.time() - start_time
                print(f"TTFT: {ttft:.2f}s")
            response += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)

    print()


async def test_websocket():
    """Test WebSocket functionality"""
    uri = "ws://localhost:8001"

    try:
        print("\n=== WebSocket Test ===")
        print("Connecting to WebSocket server...")
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending message...")

            # Send a test message
            message = {
                "messages": [
                    {"role": "user", "content": "Hello! Tell me a short joke."}
                ],
                "max_tokens": 50,
            }

            await websocket.send(json.dumps(message))
            print("Message sent. Waiting for response...")

            # Listen for response tokens
            full_response = ""
            start_time = time.time()
            first_token = False

            async for response in websocket:
                data = json.loads(response)

                if data["type"] == "token":
                    if not first_token:
                        first_token = True
                        ttft = time.time() - start_time
                        print(f"TTFT: {ttft:.2f}s")
                    content = data["content"]
                    full_response += content
                    print(content, end="", flush=True)

                elif data["type"] == "complete":
                    print(
                        f"\n\nWebSocket test complete! Full response: {full_response}"
                    )
                    break

                elif data["type"] == "error":
                    print(f"\nError: {data['message']}")
                    break

    except Exception as e:
        print(f"WebSocket connection error: {e}")
        print("Make sure the vLLM server is running with WebSocket support")


def main():
    print("=== HTTP API Test ===")
    test_http_api()

    print("\nStarting WebSocket test...")
    asyncio.run(test_websocket())


if __name__ == "__main__":
    main()
