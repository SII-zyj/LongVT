import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import SERVER_MAPPING
from torchvision.transforms.functional import to_pil_image

from data.constant import VIDEO_CAPTION_PROMPT
from data.server.openai import ChatCompletionRequest
from data.utils import encode_image, process_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--server", type=str, required=False, default="openai")
    parser.add_argument("--fps", type=int, required=False, default=1)
    parser.add_argument("--limit", type=int, required=False, default=None)
    return parser.parse_args()


def create_messages(video_path, start_time, end_time, fps):
    frames = process_video(video_path, start_time=start_time, end_time=end_time, fps=fps)
    frames = [encode_image(to_pil_image(frame)) for frame in frames]
    messages = [{"role": "user", "content": [{"type": "text", "text": VIDEO_CAPTION_PROMPT}]}]
    for frame in frames:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}}],
            }
        )
    return messages


async def generate_caption(task_list, server):
    sem = asyncio.Semaphore(cpu_count() // 2)
    messages, video_path, start_time, end_time = task_list

    async with sem:
        request = ChatCompletionRequest(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
        )
        response = await server.chat_completion_async(request)
        choices = response.choices[0]
        caption_dict = {
            "start_time": start_time,
            "caption": choices["message"]["content"],
            "end_time": end_time,
            "video_path": video_path,
        }
        return caption_dict


def generate_caption_sync(task_list, server):
    messages, video_path, start_time, end_time = task_list
    request = ChatCompletionRequest(
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        messages=messages,
        max_tokens=32768,
        temperature=0.7,
    )
    response = server.chat_completion(request)
    choices = response.choices[0]
    caption_dict = {
        "start_time": start_time,
        "caption": choices["message"]["content"],
        "end_time": end_time,
        "video_path": video_path,
    }
    return caption_dict


async def run(task_lists, server):
    tasks = [asyncio.create_task(generate_caption(task_list, server)) for task_list in task_lists]
    pbar = tqdm(total=len(tasks), desc="Generating captions")
    results = []
    for task in tasks:
        result = await task
        results.append(result)
        pbar.update(1)
    pbar.close()
    return results


def main():
    args = parse_args()
    input_path = args.input_path

    with open(input_path) as f:
        input_data = json.load(f)

    server = SERVER_MAPPING[args.server]()

    task_lists = []
    if args.limit is not None:
        input_data = input_data[: args.limit]

    with ProcessPoolExecutor(max_workers=cpu_count() // 2) as executor:
        futures = []
        info_list = []
        for item in tqdm(input_data, desc="Processing videos"):
            video_path = item["video_path"]
            start_time = item["start_time"]
            end_time = item["end_time"]
            if end_time - start_time > 10:
                # Split into 10s segments
                for i in range(0, int(end_time - start_time), 10):
                    futures.append(
                        executor.submit(
                            create_messages,
                            video_path,
                            start_time=start_time + i,
                            end_time=start_time + i + 10,
                            fps=args.fps,
                        )
                    )
                    end_time = start_time + i + 10
                    info_list.append((video_path, start_time + i, end_time))
            else:
                futures.append(
                    executor.submit(
                        create_messages, video_path, start_time=start_time, end_time=end_time, fps=args.fps
                    )
                )
                info_list.append((video_path, start_time, end_time))

        pbar = tqdm(total=len(futures), desc="Preparing tasks")
        for idx, future in enumerate(as_completed(futures)):
            task_lists.append((future.result(), *info_list[idx]))
            pbar.update(1)
        pbar.close()

    results = asyncio.run(run(task_lists, server))
    # Debugging one request
    # results = [generate_caption_sync(task_list, server) for task_list in task_lists[:1]]
    # import pdb; pdb.set_trace()
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
