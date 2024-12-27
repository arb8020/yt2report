#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "assemblyai",
#     "google-generativeai",
#     "yt-dlp",
#     "python-dotenv",
#     "rich",
#     "aiohttp",
# ]
# ///

"""
Enhanced YouTube video transcription tool using AssemblyAI and OpenRouter.
Supports downloading by channel or individual video URLs.
Organizes downloads using sanitized video titles.
Allows processing multiple videos in one go.
Intelligently resumes incomplete processing.
"""

import os
import sys
from pathlib import Path
import json
import re
from dotenv import load_dotenv
import yt_dlp
import assemblyai as aai
import aiohttp
from rich.console import Console
from rich.prompt import Prompt
import asyncio
import requests

# Load environment variables and setup
load_dotenv()
console = Console()

# Constants for OpenRouter API
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-flash-1.5-8b"

def setup_directories():
    """Create downloads directory if it doesn't exist"""
    downloads = Path("downloads")
    downloads.mkdir(exist_ok=True)
    return downloads

def sanitize_filename(name: str) -> str:
    """
    Sanitize the video title to create a valid directory name.
    Replaces spaces with underscores and removes invalid characters.
    """
    name = name.replace(' ', '_')
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name[:100]

def get_channel_videos(channel: str, limit: int = 3):
    """Get recent videos from a YouTube channel"""
    if channel.startswith('@'):
        channel = f'https://youtube.com/{channel}'

    opts = {
        'quiet': True,
        'extract_flat': True,
        'playlistreverse': True,
        'playlistend': limit
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(channel, download=False)
            channel_name = info.get('channel') or info.get('uploader')
            channel_id = info.get('channel_id') or info.get('id')
            
            playlist = ydl.extract_info(
                f'https://youtube.com/channel/{channel_id}/videos',
                download=False
            )

            videos = []
            for entry in playlist.get('entries', []):
                if entry:
                    videos.append({
                        'id': entry['id'],
                        'title': entry['title'],
                        'url': f"https://youtube.com/watch?v={entry['id']}",
                        'channel': info.get('channel') or info.get('uploader'),
                        'channel_id': info.get('channel_id') or info.get('uploader_id')
                    })

            return channel_name, videos

        except Exception as e:
            console.print(f"[red]Error getting channel videos: {e}[/red]")
            sys.exit(1)

def get_video_info(video_url: str):
    """Extract video information from a YouTube URL"""
    opts = {
        'quiet': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
            return {
                'id': info.get('id'),
                'title': info.get('title'),
                'url': info.get('webpage_url'),
                'channel': info.get('channel') or info.get('uploader'),
                'channel_id': info.get('channel_id') or info.get('uploader_id')
            }
        except Exception as e:
            console.print(f"[red]Error getting video info: {e}[/red]")
            return None

def download_audio(video_id: str, output_dir: Path):
    """Download video audio"""
    opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            url = f'https://youtube.com/watch?v={video_id}'
            ydl.download([url])
            return output_dir / f"{video_id}.mp3"
    except Exception as e:
        console.print(f"[red]Error downloading audio: {e}[/red]")
        return None

def transcribe(audio_path: Path, api_key: str):
    """Get transcript from AssemblyAI"""
    try:
        aai.settings.api_key = api_key
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(str(audio_path), config=config)
        return transcript.text
    except Exception as e:
        console.print(f"[red]Error getting transcript: {e}[/red]")
        return None

async def enhance_transcript(transcript: str, api_key: str):
    """Enhance transcript using Google's Gemini"""
    PROMPT = """You are an expert transcript editor. Your task is to enhance this transcript for maximum readability while maintaining the core message.
IMPORTANT: Respond ONLY with the enhanced transcript. Do not include any explanations, headers, or phrases like "Here is the transcript."

Think about your job as if you were transcribing an interview for a print book where the priority is the reading audience. It should be a pleasure to read this as a written artifact where all the flubs, repetitions, conversational artifacts, and filler words are removed.

Please:
1. Fix speaker attribution errors and incomplete thoughts
2. Optimize AGGRESSIVELY for readability:
   * Remove ALL conversational artifacts (yeah, so, I mean, etc.)
   * Remove ALL filler words (um, uh, like, you know)
   * Remove false starts and self-corrections
   * Remove redundant phrases and hesitations
   * Convert indirect or rambling responses into direct statements
   * Break up run-on sentences into clear statements
3. Format consistently:
   * Keep the "Speaker X 00:00:00" format
   * Add TWO line breaks between speaker/timestamp and text
   * Use proper punctuation and capitalization
   * Add paragraph breaks for topic changes
   * Don't go more than four sentences without a paragraph break"""

    try:
        generativeai.configure(api_key=api_key)
        model = generativeai.GenerativeModel("gemini-exp-1206")
        response = await model.generate_content_async([PROMPT, transcript])
        return response.text
    except Exception as e:
        console.print(f"[red]Error enhancing transcript: {e}[/red]")
        return None

def check_processing_status(video_dir: Path) -> dict:
    """Check which processing steps need to be done"""
    return {
        'audio_exists': any(video_dir.glob('*.mp3')),
        'raw_transcript_exists': (video_dir / "transcript_raw.txt").exists(),
        'enhanced_transcript_exists': (video_dir / "transcript_enhanced.md").exists(),
        'structured_report_exists': (video_dir / "transcript_report.md").exists()
    }

def create_structured_report(transcript: str, api_key: str):
    """Transform enhanced transcript into structured guide using OpenRouter API"""
    
    SYSTEM_PROMPT = """You are an expert content structuring assistant. Your role is to transform video transcripts into clear, actionable guides while preserving all important context and details. Always provide complete, thorough responses."""

    USER_PROMPT = """Transform this transcript into a structured guide following this exact format:

# [Title]

## Quick Reference
* Key metrics (time, difficulty, cost, etc.)
* Core requirements
* Basic overview

## Core Content
[Most immediately actionable content goes here, organized by relevance]
* Essential steps
* Key ingredients/components
* Critical timings
* Basic variations

---

## Supporting Information
[Context and additional details go here]
* Historical background
* Detailed tips
* Common mistakes
* Extended variations
* Expert insights

Guidelines:
1. ALWAYS put immediately actionable content before context/background
2. Use clear, concise headers
3. Maintain consistent formatting
4. Group related information
5. Highlight critical steps/components
6. Preserve important details from original
7. Remove conversational artifacts
8. Add paragraph breaks for readability
9. Process the ENTIRE transcript
10. Include ALL relevant information

Example action verbs for steps: Prepare, Combine, Heat, Add, Mix, etc.
Example time indicators: Immediately, After X minutes, Until golden, etc.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "user", "content": transcript}
        ],
        "model": MODEL,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            console.print(f"[red]Error creating structured report: {response.status_code}[/red]")
            console.print(f"Error details: {response.text}")
            return None
    except Exception as e:
        console.print(f"[red]Error creating structured report: {e}[/red]")
        return None

async def process_video(video: dict, channel_dir: Path, assemblyai_key: str, google_key: str, openrouter_key: str):
    """Process a single video"""
    console.print(f"\nProcessing: {video['title']}")

    sanitized_title = sanitize_filename(video['title'])
    video_dir = channel_dir / sanitized_title
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Check what needs to be done
    status = check_processing_status(video_dir)
    
    if all(status.values()):
        console.print(f"[yellow]Skipping '{video['title']}' - already fully processed.[/yellow]")
        return

    # Step 1: Download audio if needed
    audio_path = None
    if not status['audio_exists']:
        console.print("Downloading audio...")
        audio_path = download_audio(video['id'], video_dir)
        if not audio_path or not audio_path.exists():
            console.print(f"[red]Failed to download audio for '{video['title']}'[/red]")
            return
    else:
        audio_path = next(video_dir.glob('*.mp3'))
        console.print("[green]Using existing audio file.[/green]")

    # Step 2: Generate raw transcript if needed
    if not status['raw_transcript_exists']:
        console.print("Generating transcript...")
        transcript = transcribe(audio_path, assemblyai_key)
        if transcript:
            (video_dir / "transcript_raw.txt").write_text(transcript)
        else:
            console.print(f"[red]Failed to generate transcript for '{video['title']}'[/red]")
            return
    else:
        console.print("[green]Using existing raw transcript.[/green]")

    # Step 3: Enhance transcript if needed
    if not status['enhanced_transcript_exists']:
        console.print("Enhancing transcript...")
        raw_transcript = (video_dir / "transcript_raw.txt").read_text()
        enhanced = await enhance_transcript(raw_transcript, google_key)
        if enhanced:
            (video_dir / "transcript_enhanced.md").write_text(enhanced)
            console.print("[green]Enhanced transcript generated.[/green]")
        else:
            console.print("[red]Failed to enhance transcript - saving raw version.[/red]")
            return
    else:
        console.print("[green]Enhanced transcript already exists.[/green]")

    # Step 4: Create structured report if needed
    if not status['structured_report_exists']:
        console.print("Creating structured report...")
        enhanced_transcript = (video_dir / "transcript_enhanced.md").read_text()
        structured_report = create_structured_report(enhanced_transcript, openrouter_key)
        if structured_report:
            (video_dir / "transcript_report.md").write_text(structured_report)
            console.print("[green]Structured report generated.[/green]")
        else:
            console.print("[red]Failed to create structured report.[/red]")
            return
    else:
        console.print("[green]Structured report already exists.[/green]")

    # Save metadata if it doesn't exist
    if not (video_dir / "metadata.json").exists():
        (video_dir / "metadata.json").write_text(json.dumps(video, indent=2))

    console.print(f"[green]✓ Processing complete for: {video['title']}[/green]")

async def process_videos(videos, channel_dir, assemblyai_key, google_key, openrouter_key):
    """Process multiple videos concurrently"""
    tasks = [
        process_video(video, channel_dir, assemblyai_key, google_key, openrouter_key)
        for video in videos
    ]
    await asyncio.gather(*tasks)

def main():
    # Check API keys
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    assemblyai_key = os.getenv('ASSEMBLYAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    if not all([assemblyai_key, google_key, openrouter_key]):
        console.print("[red]Error: Missing API keys in .env file[/red]")
        console.print("Required: ASSEMBLYAI_API_KEY, GOOGLE_API_KEY, and OPENROUTER_API_KEY")
        sys.exit(1)

    # User choice: Channel or Video
    console.print("Choose download mode:")
    console.print("1. By Channel")
    console.print("2. By Video URL")
    choice = Prompt.ask("Enter choice (1 or 2)", choices=["1", "2"], default="1")

    output_dir = setup_directories()

    if choice == "1":
        # Download by Channel
        channel_input = Prompt.ask("Enter channel handle (e.g., @channel) or channel URL")
        limit_input = Prompt.ask("Enter number of recent videos to process", default="3")
        try:
            limit = int(limit_input)
        except ValueError:
            console.print("[red]Invalid number. Using default value 3.[/red]")
            limit = 3

        console.print(f"\nFetching videos from {channel_input}...")
        channel_name, videos = get_channel_videos(channel_input, limit)

        if not videos:
            console.print("[red]No videos found.[/red]")
            sys.exit(1)

        # Show videos
        console.print("\nAvailable videos:")
        for i, video in enumerate(videos, 1):
            console.print(f"{i}. {video['title']}")

        # Get selection
        selection = Prompt.ask("\nEnter video numbers to process (e.g., 1,3,5 or 1-5), or press Enter for all", default="")
        if selection.strip():
            try:
                selected = []
                for part in selection.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected.extend(range(start, end + 1))
                    else:
                        selected.append(int(part.strip()))
                selected_videos = [videos[i-1] for i in selected if 0 < i <= len(videos)]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection. Exiting.[/red]")
                sys.exit(1)
        else:
            selected_videos = videos

        # Process videos
        channel_dir = output_dir / sanitize_filename(channel_name)
        channel_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(process_videos(selected_videos, channel_dir, assemblyai_key, google_key, openrouter_key))

    elif choice == "2":
        # Handle individual video URLs
        console.print("\nChoose input method:")
        console.print("1. Enter URLs manually")
        console.print("2. Read from file")
        input_choice = Prompt.ask("Enter choice", choices=["1", "2"], default="1")

        video_urls = []
        if input_choice == "1":
            while True:
                url = Prompt.ask("Enter YouTube URL (or press Enter to finish)", default="")
                if not url.strip():
                    break
                video_urls.append(url.strip())
        else:
            file_path = Prompt.ask("Enter path to file with URLs (one per line)")
            try:
                with open(file_path) as f:
                    video_urls = [line.strip() for line in f if line.strip()]
            except Exception as e:
                console.print(f"[red]Error reading file: {e}[/red]")
                sys.exit(1)

        if not video_urls:
            console.print("[red]No URLs provided. Exiting.[/red]")
            sys.exit(1)

        # Process URLs
        videos = []
        for url in video_urls:
            if video := get_video_info(url):
                videos.append(video)

        if not videos:
            console.print("[red]No valid videos found. Exiting.[/red]")
            sys.exit(1)

        # Organize by channel
        for video in videos:
            channel_name = video['channel'] or "unknown_channel"
            channel_dir = output_dir / sanitize_filename(channel_name)
            channel_dir.mkdir(parents=True, exist_ok=True)
            asyncio.run(process_video(video, channel_dir, assemblyai_key, google_key, openrouter_key))

    console.print("\n[green]Processing complete![/green]")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
