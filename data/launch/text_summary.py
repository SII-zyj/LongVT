#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Summary Module

This module performs hierarchical (bottom-up) multi-round summarization of dense video captions.
It reads video caption JSON files and generates concise, structured summaries.

Workflow:
  1. Read raw JSON format: [{"start_time": X, "caption": "...", "end_time": Y}]
  2. Auto-group: Group raw segments by specified size (default 8)
  3. R0: Summarize each group's merged captions -> concise lines:
      "Video_event X(+Y+...) [start – end]: <summary>"
  4. R1..N: Pairwise merge previous round summaries, maintaining:
      - Chronological order
      - Event ID set labels
      - Merged time ranges (min start, max end)
      - Conciseness and global coherence

Usage:
    python launch/text_summary.py --input-dir /path/to/captions --output-file /path/to/output.json
"""

import json
import os
import logging
import time
import math
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
from tqdm import tqdm


# System prompts for summarization
SYSTEM_SUMMARIZE = """You are a precise summarizer for dense video captions.
You must output a SHORTER, COARSE-GRAINED, chronological summary that still preserves:
- Event IDs and timestamp ranges
- If you merge related events, combine their ID labels with '+' and merge times to [min_start – max_end]
- Always keep lines in order and self-contained, providing global context, not frame-by-frame details.
- Strongly compress repetitive details and remove markdown noise.
- Focus on main actions, actors, and context."""

USER_INSTR_SUMMARIZE = """INPUT:
You are given ONE item's `merged_caption` which includes many lines like:
  Video_event <ID> [<start>s - <end>s]: <dense text...>

TASK:
1) Read all events in this merged_caption.
2) Produce a concise, coarse-grained sequence of lines that summarize the whole item,
   merging obviously related sub-events.
3) Preserve event IDs and merged times. If merging multiple events, join IDs with '+'.

OUTPUT FORMAT (strict):
One line per summarized interval:
Video_event <ID or ID+ID+...> [<start_time>s – <end_time>s]: <Concise summary>

Rules:
- Keep chronological order and global context.
- Avoid markdown headings or bullet points.
- No extra text before or after the lines."""

SYSTEM_MERGE = """You are a careful merger of two already-summarized caption lists.
You must return a SINGLE, SHORTER, chronological sequence that:
- Merges adjacent or overlapping intervals that describe the same scene/subject/theme.
- When merging, combine Event IDs using '+' and merge timestamps to [min_start – max_end].
- Preserve self-contained global context. Remove redundancies.
- Keep format EXACTLY as specified. Do not add commentary."""

USER_INSTR_MERGE = """INPUT:
Two sequences of lines in the exact format:
Video_event ... [start – end]: ...
Video_event ... [start – end]: ...
(Sequence A)
---
(Sequence B)

TASK:
Return ONE merged, concise, chronological sequence. Merge adjacent/overlapping entries when appropriate.

OUTPUT FORMAT (strict):
Video_event <ID or ID+ID+...> [<start_time>s – <end_time>s]: <Concise summary>

Rules:
- Keep only the lines, nothing else.
- Maintain ascending time order.
- Merge times correctly (min start, max end).
- Combine IDs with '+' when merging.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Safety margin for long inputs
MAX_CHARS_PER_ITEM = 50000


@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RoundStats:
    """Statistics for a single round."""
    round_index: int
    api_calls: int = 0
    usage: Usage = field(default_factory=Usage)


@dataclass
class RunStats:
    """Overall run statistics."""
    rounds: List[RoundStats] = field(default_factory=list)

    def add_usage(self, rnd: int, u_prompt: int, u_completion: int, u_total: int):
        while len(self.rounds) <= rnd:
            self.rounds.append(RoundStats(round_index=len(self.rounds)))
        self.rounds[rnd].api_calls += 1
        self.rounds[rnd].usage.prompt_tokens += u_prompt
        self.rounds[rnd].usage.completion_tokens += u_completion
        self.rounds[rnd].usage.total_tokens += u_total

    def totals(self) -> Usage:
        u = Usage()
        for r in self.rounds:
            u.prompt_tokens += r.usage.prompt_tokens
            u.completion_tokens += r.usage.completion_tokens
            u.total_tokens += r.usage.total_tokens
        return u


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Video Caption Summary Tool - Using OpenAI API for caption summarization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic usage
  python text_summary.py --input-dir /path/to/captions --output-file /path/to/output.json
  
  # With sharding for parallel processing
  python text_summary.py --input-dir /path/to/captions --output-file /path/to/output.json --shard-id 0 --total-shards 4
  
  # Custom group size
  python text_summary.py --input-dir /path/to/captions --output-file /path/to/output.json --group-size 10
        """
    )
    
    parser.add_argument('--model', default='gpt-4o', choices=['gpt-4o', 'o3'],
                        help='OpenAI model name (default: gpt-4o)')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory containing caption JSON files')
    parser.add_argument('--output-file', required=True,
                        help='Output JSON file for video summaries')
    parser.add_argument('--group-size', type=int, default=8,
                        help='Group size for long videos (>16 segments). Adaptive: <8 no grouping, 8-16 use 4, >16 use this value (default: 8)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--shard-id', type=int, default=None,
                        help='Current shard ID (0-based)')
    parser.add_argument('--total-shards', type=int, default=1,
                        help='Total number of shards (default: 1)')
    
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    return args


def init_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')
        
        if not api_key:
            logging.error("OPENAI_API_KEY environment variable not set")
            return None
        
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            
        client = OpenAI(**kwargs)
        logging.info("OpenAI API client initialized successfully")
        return client
            
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None


def find_all_caption_files(base_dir: str) -> List[str]:
    """Find all caption JSON files in the specified directory."""
    caption_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                caption_files.append(os.path.join(root, file))
    logging.info(f"Found {len(caption_files)} caption files in {base_dir}")
    return sorted(caption_files)


def load_caption_file(input_file: str) -> Optional[List[Dict]]:
    """Load caption file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logging.error("Data format error: expected list format")
            return None
        
        # Validate required fields
        if data and not all(key in data[0] for key in ["start_time", "end_time", "caption"]):
            logging.error("Data format error: missing required fields (start_time, end_time, caption)")
            return None
            
        logging.info(f"Successfully loaded {len(data)} caption records")
        return data
        
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


def get_video_id_from_path(file_path: str) -> str:
    """Extract video ID from file path (filename without extension)."""
    return Path(file_path).stem


def get_adaptive_group_size(caption_count: int, base_group_size: int = 8) -> int:
    """
    Determine adaptive group size based on caption count.
    
    Args:
        caption_count: Number of caption segments
        base_group_size: Base group size for long videos
        
    Returns:
        Appropriate group size
    """
    if caption_count < 8:
        return caption_count  # Short video, no grouping
    elif caption_count <= 16:
        return 4  # Medium video, small groups
    else:
        return base_group_size  # Long video, use specified group size


def process_captions_to_groups(captions: List[Dict], group_size: int) -> List[Dict]:
    """
    Process captions into groups for summarization.
    
    Args:
        captions: List of caption dictionaries
        group_size: Base group size
        
    Returns:
        List of group dictionaries with merged captions
    """
    # Sort by start time
    captions.sort(key=lambda x: x["start_time"])
    
    # Get adaptive group size
    adaptive_group_size = get_adaptive_group_size(len(captions), group_size)
    
    groups = []
    for i in range(0, len(captions), adaptive_group_size):
        group_segments = captions[i:i + adaptive_group_size]
        
        # Merge captions in this group
        merged_caption_parts = []
        for j, segment in enumerate(group_segments):
            event_id = i + j + 1
            time_info = f"[{segment['start_time']:.1f}s - {segment['end_time']:.1f}s]"
            caption_with_time = f"Video_event {event_id} {time_info}: {segment['caption']}"
            merged_caption_parts.append(caption_with_time)
        
        merged_caption = "\n\n".join(merged_caption_parts)
        
        group_info = {
            "group_id": len(groups) + 1,
            "start_segment": i + 1,
            "end_segment": i + len(group_segments),
            "segments_count": len(group_segments),
            "group_start_time": group_segments[0]["start_time"],
            "group_end_time": group_segments[-1]["end_time"],
            "merged_caption": merged_caption
        }
        groups.append(group_info)
    
    return groups


def maybe_truncate(text: str, max_chars: int) -> str:
    """Truncate text if it exceeds max characters."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def parse_summary_lines(text: str) -> List[str]:
    """Parse and validate summary lines."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    pat = re.compile(r"^Video_event\s+.+\[\s*[\d\.]+s\s*[–-]\s*[\d\.]+s\]:\s+.+$")
    return [ln for ln in lines if pat.match(ln)]


def merge_pairwise(seq: List[str]) -> List[str]:
    """Parse lines into normalized format and sort by time."""
    parsed = []
    pat = re.compile(
        r"^Video_event\s+(.+?)\s*\[\s*([\d\.]+)s\s*[–-]\s*([\d\.]+)s\]\s*:\s*(.+)$"
    )
    for ln in seq:
        m = pat.match(ln)
        if not m:
            continue
        ids = m.group(1).strip()
        start = float(m.group(2))
        end = float(m.group(3))
        txt = m.group(4).strip()
        parsed.append((ids, start, end, txt))
    
    # Sort by start time
    parsed.sort(key=lambda x: (x[1], x[2]))
    
    # Return normalized lines
    out = []
    for ids, start, end, txt in parsed:
        out.append(f"Video_event {ids} [{start:.1f}s – {end:.1f}s]: {txt}")
    return out


def call_summarize(client: OpenAI, merged_caption: str, model: str = "gpt-4o") -> Tuple[Optional[str], Usage]:
    """Call LLM to summarize merged caption."""
    try:
        merged_caption = maybe_truncate(merged_caption, MAX_CHARS_PER_ITEM)
        user_msg = f"{USER_INSTR_SUMMARIZE}\n\n---\n{merged_caption}"
        
        if model == "o3":
            resp = client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": SYSTEM_SUMMARIZE},
                    {"role": "user", "content": user_msg}
                ],
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_SUMMARIZE},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
        
        content = resp.choices[0].message.content.strip()
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        
        lines = parse_summary_lines(content)
        return "\n".join(lines), usage
        
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None, Usage()


def call_merge(client: OpenAI, text_a: str, text_b: str, model: str = "gpt-4o") -> Tuple[Optional[str], Usage]:
    """Call LLM to merge two summary sequences."""
    try:
        a_lines = parse_summary_lines(text_a)
        b_lines = parse_summary_lines(text_b)
        a_norm = "\n".join(merge_pairwise(a_lines))
        b_norm = "\n".join(merge_pairwise(b_lines))
        
        user_msg = f"{USER_INSTR_MERGE}\n\n(Sequence A)\n{a_norm}\n---\n(Sequence B)\n{b_norm}"
        
        if model == "o3":
            resp = client.chat.completions.create(
                model="o3",
                messages=[
                    {"role": "system", "content": SYSTEM_MERGE},
                    {"role": "user", "content": user_msg}
                ],
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MERGE},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
        
        content = resp.choices[0].message.content.strip()
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
        
        merged_lines = parse_summary_lines(content)
        merged_lines = merge_pairwise(merged_lines)
        return "\n".join(merged_lines), usage
        
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None, Usage()


def round0_group_summaries(client: OpenAI, items: List[Dict], stats: RunStats, model: str) -> List[str]:
    """
    Round 0: Summarize each item's merged_caption into concise lines.
    """
    outputs = []
    for idx, it in enumerate(tqdm(items, desc="Round 0: summarizing groups", unit="group")):
        merged_caption = it["merged_caption"]
        text, usage = call_summarize(client, merged_caption, model=model)
        stats.add_usage(0, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        
        if text:
            outputs.append(text)
        else:
            outputs.append("")
    
    return outputs


def iterative_merge_rounds(client: OpenAI, initial_blocks: List[str], stats: RunStats, model: str) -> str:
    """
    Repeatedly merge blocks two-by-two until one remains.
    """
    round_idx = 1
    current = initial_blocks[:]
    
    while len(current) > 1:
        logging.info(f"Round {round_idx}: merging {len(current)} block(s) into ~{math.ceil(len(current)/2)}")
        
        next_blocks = []
        iterator = range(0, len(current), 2)
        iterator = tqdm(iterator, total=math.ceil(len(current)/2), 
                       desc=f"Round {round_idx}: pairwise merge", unit="pair")
        
        for i in iterator:
            if i + 1 < len(current):
                merged, usage = call_merge(client, current[i], current[i+1], model=model)
                stats.add_usage(round_idx, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
                next_blocks.append(merged if merged else "")
            else:
                # Carry forward odd one
                next_blocks.append(current[i])
        
        current = next_blocks
        round_idx += 1
    
    return current[0] if current else ""


def process_single_video(
    client: OpenAI,
    input_file: str,
    stats: RunStats,
    model: str = "gpt-4o",
    group_size: int = 8
) -> Optional[Dict]:
    """Process a single video's caption file and generate summary."""
    video_id = get_video_id_from_path(input_file)
    logging.info(f"Processing video: {video_id}")
    
    # Load captions
    captions = load_caption_file(input_file)
    if not captions:
        logging.warning(f"Failed to load captions for {video_id}")
        return None
    
    # Record initial token usage
    initial_usage = stats.totals()
    
    try:
        # Process captions into groups
        items = process_captions_to_groups(captions, group_size)
        logging.info(f"Created {len(items)} groups from {len(captions)} captions")
        
        # Round 0: summarize each group
        r0_blocks = round0_group_summaries(client, items, stats, model)
        logging.info(f"Round 0 complete: generated {len(r0_blocks)} group summaries")
        
        # Subsequent rounds: iterative pairwise merging
        final_text = iterative_merge_rounds(client, r0_blocks, stats, model)
        
        # Calculate token usage for this video
        final_usage = stats.totals()
        video_usage = Usage(
            prompt_tokens=final_usage.prompt_tokens - initial_usage.prompt_tokens,
            completion_tokens=final_usage.completion_tokens - initial_usage.completion_tokens,
            total_tokens=final_usage.total_tokens - initial_usage.total_tokens
        )
        
        logging.info(f"Video {video_id} complete. Tokens: {video_usage.total_tokens}")
        
        return {
            "video_id": video_id,
            "summary": final_text,
            "segments_processed": len(captions),
            "groups_created": len(items),
            "token_usage": {
                "prompt_tokens": video_usage.prompt_tokens,
                "completion_tokens": video_usage.completion_tokens,
                "total_tokens": video_usage.total_tokens
            }
        }
        
    except Exception as e:
        logging.error(f"Error processing video {video_id}: {e}")
        return {
            "video_id": video_id,
            "summary": "",
            "error": str(e),
            "segments_processed": 0,
            "groups_created": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


def save_results(results: List[Dict], output_file: str, model: str) -> bool:
    """Save all results to output file."""
    try:
        # Calculate totals
        total_tokens = sum(r.get('token_usage', {}).get('total_tokens', 0) for r in results)
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        final_data = {
            "summary_metadata": {
                "total_videos": len(results),
                "successful_videos": successful,
                "failed_videos": failed,
                "total_tokens_used": total_tokens,
                "model_used": model,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "video_summaries": results
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Results saved to: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        return False


def load_existing_results(output_file: str) -> Tuple[List[Dict], set]:
    """Load existing results for resume capability."""
    if not os.path.exists(output_file):
        return [], set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'video_summaries' in data:
            existing_results = data['video_summaries']
        else:
            existing_results = data if isinstance(data, list) else []
        
        processed_ids = {r.get('video_id', '') for r in existing_results if r.get('video_id')}
        
        logging.info(f"Found existing results: {len(processed_ids)} videos already processed")
        return existing_results, processed_ids
        
    except Exception as e:
        logging.warning(f"Error loading existing results: {e}")
        return [], set()


def main():
    """Main function - batch process all caption files."""
    args = parse_arguments()
    
    client = init_openai_client()
    if not client:
        logging.error("API client initialization failed, exiting")
        return
    
    # Find all caption files
    caption_files = find_all_caption_files(args.input_dir)
    if not caption_files:
        logging.error(f"No caption files found in {args.input_dir}")
        return
    
    # Apply sharding if specified
    if args.shard_id is not None:
        total_files = len(caption_files)
        files_per_shard = total_files // args.total_shards
        start_idx = args.shard_id * files_per_shard
        if args.shard_id == args.total_shards - 1:
            end_idx = total_files
        else:
            end_idx = start_idx + files_per_shard
        
        caption_files = caption_files[start_idx:end_idx]
        logging.info(f"Shard {args.shard_id} processing files {start_idx+1}-{end_idx} ({len(caption_files)} files)")
    
    # Load existing results for resume
    existing_results, processed_ids = load_existing_results(args.output_file)
    all_results = existing_results.copy()
    
    stats = RunStats()
    success_count = 0
    skipped_count = 0
    total_files = len(caption_files)
    
    for i, input_file in enumerate(tqdm(caption_files, desc="Processing caption files", unit="file"), 1):
        try:
            video_id = get_video_id_from_path(input_file)
            
            # Skip if already processed
            if video_id in processed_ids:
                logging.info(f"Skipping file {i}/{total_files}: {video_id} (already processed)")
                skipped_count += 1
                continue
            
            logging.info(f"Processing file {i}/{total_files}: {video_id}")
            
            result = process_single_video(
                client, input_file, stats, 
                model=args.model, 
                group_size=args.group_size
            )
            
            if result:
                all_results.append(result)
                success_count += 1
                
                # Save after each video for resume capability
                save_results(all_results, args.output_file, args.model)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logging.error(f"Error processing file {input_file}: {e}")
            continue
    
    # Final save
    save_results(all_results, args.output_file, args.model)
    
    # Print summary
    logging.info(f"\nBatch summarization complete")
    logging.info(f"Total files: {total_files}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Skipped (already processed): {skipped_count}")
    logging.info(f"Failed: {total_files - success_count - skipped_count}")
    logging.info(f"Output file: {args.output_file}")
    
    # Print token usage
    totals = stats.totals()
    logging.info(f"Total tokens used: {totals.total_tokens}")


if __name__ == "__main__":
    main()

