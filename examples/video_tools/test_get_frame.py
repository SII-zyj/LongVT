#!/usr/bin/env python3
"""
Test script for the get_frame MCP tool.

Usage:
  python examples/video_tools/test_get_frame.py \
    --video_path /path/to/video.mp4 \
    --timestamp 3.5 \
    --output_dir /tmp/get_frame_frames
"""

import argparse
import asyncio
import base64
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_get_frame(server_path: str, video_path: str, timestamp: float, output_dir: str) -> None:
    server_params = StdioServerParameters(command="python", args=[server_path])
    async with stdio_client(server=server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("get_frame", {"video_path": video_path, "timestamp": timestamp})

    text_parts = [part.text for part in result.content if part.type == "text"]
    image_parts = [part for part in result.content if part.type == "image"]
    if not image_parts:
        error_text = "\n".join(text_parts).strip()
        if error_text:
            raise RuntimeError(f"get_frame tool returned no image content: {error_text}")
        raise RuntimeError("No image content returned from get_frame tool.")

    os.makedirs(output_dir, exist_ok=True)
    for index, image_part in enumerate(image_parts):
        image_data = base64.b64decode(image_part.data)
        output_path = os.path.join(output_dir, f"frame_{index + 1}.png")
        with open(output_path, "wb") as f:
            f.write(image_data)

        print(f"Saved frame {index + 1} to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test get_frame MCP tool.")
    parser.add_argument("--video_path", required=True, help="Path to the video file.")
    parser.add_argument("--timestamp", type=float, required=True, help="Timestamp in seconds.")
    parser.add_argument(
        "--server_path",
        default="examples/video_tools/mcp_server.py",
        help="Path to the MCP server script.",
    )
    parser.add_argument(
        "--output_dir",
        default="get_frame_outputs",
        help="Directory to save the extracted frames.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video path is not a file: {args.video_path}")

    asyncio.run(run_get_frame(args.server_path, args.video_path, args.timestamp, args.output_dir))


if __name__ == "__main__":
    main()
