# Copyright 2025 Individual Contributor: Kaichen Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
from io import BytesIO

import numpy as np
from decord import VideoReader, cpu
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image

app = FastMCP("Video Tools MCP Server", "0.1.0")


@app.tool(name="crop_video", description="Crop a video to a specified duration.")
def crop_video(
    video_path: Annotated[str, Field(description="Path to the video file")],
    start_time: Annotated[float, Field(description="Start time in seconds")],
    end_time: Annotated[float, Field(description="End time in seconds, must be > start_time")]
) -> list[ImageContent]:
    """
    Crop a video to a specified duration.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        str: Path to the cropped video file.
    """
    # Placeholder for actual video cropping logic
    nframes = 16
    vr = VideoReader(video_path, ctx=cpu(0))
    start_frame = int(start_time * vr.get_avg_fps())
    end_frame = int(end_time * vr.get_avg_fps())
    uniform_sampled_frames = np.linspace(start_frame, end_frame - 1, nframes, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # Turn np into base64 encoded
    images = [Image.fromarray(frame) for frame in spare_frames]
    # Encode images to base64
    image_contents = []
    for img in images:
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        image_contents.append(ImageContent(type="image", data=base64_str, mimeType="image/png"))

    return image_contents

# video_zoom_in_tool
# @app.tool(
#     name="video_zoom_in_tool",
#     description="Crops a region (bbox_2d) from a single frame specified by timestamp_sec in a video."
# )
# def video_zoom_in_tool(
#     video_path: Annotated[str, Field(description="Path to the video file")],
#     timestamp_sec: Annotated[float, Field(description="Timestamp in seconds")],
#     bbox_2d: Annotated[list[float], Field(description="[x1, y1, x2, y2] in pixels")]
# ) -> ImageContent:

#     if len(bbox_2d) != 4:
#         raise ValidationError("bbox_2d must be [x1, y1, x2, y2].")
#     x1, y1, x2, y2 = map(int, bbox_2d)
#     if x2 <= x1 or y2 <= y1:
#         raise ValidationError("bbox_2d invalid: require x2>x1 and y2>y1.")

#     path = Path(video_path)
#     if not path.exists():
#         raise FileNotFoundError(f"{path} not found") 

#     vr  = VideoReader(str(path), ctx=cpu(0))          
#     fps = vr.get_avg_fps()                           
#     frame_idx = int(timestamp_sec * fps)

#     frame_nd = vr.get_batch([frame_idx]).asnumpy()[0] 
#     img = Image.fromarray(frame_nd).convert("RGB")   
#     cropped = img.crop((x1, y1, x2, y2))

#     buf = BytesIO()
#     cropped.save(buf, format="PNG")
#     b64 = base64.b64encode(buf.getvalue()).decode("utf-8")  

#     return ImageContent(type="image", data=b64, mimeType="image/png")

if __name__ == "__main__":
    app.run()