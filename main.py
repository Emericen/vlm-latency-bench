#!/usr/bin/env python3
"""
Simple approach: Start standard vLLM server and add WebSocket client
"""

import os
import asyncio
import json
import websockets
from openai import OpenAI


async def websocket_client():
    """WebSocket client that connects to HTTP API"""

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    async def handle_websocket(websocket):
        print(f"WebSocket client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                data = json.loads(message)
                print(f"Received message: {data}")

                # Forward to HTTP API
                try:
                    response = client.chat.completions.create(
                        model=data.get("model", "Qwen/Qwen2.5-VL-7B-Instruct"),
                        messages=data.get("messages", []),
                        max_tokens=data.get("max_tokens", 100),
                        stream=True,
                    )

                    # Stream response back
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "token",
                                        "content": chunk.choices[0].delta.content,
                                    }
                                )
                            )

                    await websocket.send(json.dumps({"type": "complete"}))

                except Exception as e:
                    await websocket.send(
                        json.dumps({"type": "error", "message": str(e)})
                    )

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket client disconnected")

    # Start WebSocket server on different port
    print("Starting WebSocket server on port 8001...")
    server = await websockets.serve(handle_websocket, "0.0.0.0", 8001)
    print("WebSocket server running on ws://localhost:8001")
    await server.wait_closed()


if __name__ == "__main__":
    print("This is a WebSocket proxy for vLLM HTTP API")
    print("1. Start vLLM server: vllm serve Qwen/Qwen2.5-VL-7B-Instruct")
    print("2. Run this script to add WebSocket support")
    print("3. Connect to ws://localhost:8001")

    asyncio.run(websocket_client())
