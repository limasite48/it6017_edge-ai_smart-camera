"""
IT6017 - Digital Transformation: Edge AI Smart Camera System
Dataset Preparation: Download and verify edge AI benchmark surveillance test videos.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
import urllib.request
import cv2
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

TEST_VIDEOS: Dict[str, Dict[str, str]] = {
    "one-by-one-person-detection.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/one-by-one-person-detection.mp4",
        "description": "Single-person doorway ingress/egress surveillance scenario",
        "category": "Single Person / Doorway",
    },
    "people-detection.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4",
        "description": "Multi-person corridor and public hallway surveillance scenario",
        "category": "Multi-Person / Corridor",
    },
    "face-demographics-walking.mp4": {
        "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
        "description": "Approaching pedestrian in commercial office hallway",
        "category": "Head-on Pedestrian",
    },
    "object-detection-01.mp4": {
        "url": "",
        "description": "Big Buck Bunny animated film sequence featuring animals and nature objects",
        "category": "Animation / General Objects",
    },
    "object-detection-02.mp4": {
        "url": "",
        "description": "Vertical 4K smartphone capture (9:16) of bustling street intersection with cars and pedestrians",
        "category": "Vertical 4K / Urban Traffic",
    },
    "object-detection-03.mp4": {
        "url": "",
        "description": "Horizontal 4K CCTV surveillance camera overlooking crossroad with vehicles and signage",
        "category": "Horizontal 4K / CCTV Traffic",
    },
}


def download_video_with_progress(url: str, dest_path: Path) -> bool:
    """Download video with interactive Rich progress bar or verify user-provided file."""
    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        source_label = "user-provided" if not url else "cached"
        console.print(f"[green]Using {source_label} file:[/green] {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    if not url:
        console.print(f"[bold red]File not found: {dest_path.name} (no download URL available)[/bold red]")
        return False

    temp_path = dest_path.with_suffix(".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("download", filename=dest_path.name, start=False)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "IT6017-EdgeAI-Benchmark/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "wb") as out_file:
                total_size = int(response.info().get("Content-Length", 0))
                progress.update(task_id, total=total_size)
                progress.start_task(task_id)

                block_size = 64 * 1024
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    progress.update(task_id, advance=len(buffer))

            temp_path.replace(dest_path)
            return True
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            console.print(f"[bold red]Failed to download {dest_path.name}: {e}[/bold red]")
            return False


def verify_video(video_path: Path) -> Dict[str, Any]:
    """Verify video can be decoded by OpenCV and extract metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"valid": False, "error": "OpenCV cannot open video stream"}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / max(1e-5, fps)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {"valid": False, "error": "Failed to read first frame"}

    return {
        "valid": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "size_mb": video_path.stat().st_size / (1024 * 1024),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify edge AI benchmark surveillance videos.")
    parser.add_argument("--output-dir", default="data/test_videos", help="Destination folder (default: data/test_videos)")
    parser.add_argument("--video", choices=list(TEST_VIDEOS.keys()) + ["all"], default="all", help="Specific video to download")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = TEST_VIDEOS.keys() if args.video == "all" else [args.video]

    console.print(f"[bold cyan]Preparing Edge AI Benchmark Test Videos in:[/bold cyan] {out_dir}\n")

    summary_table = Table(title="Test Videos Catalogue & Integrity Verification", show_lines=True)
    summary_table.add_column("Filename", style="bold green")
    summary_table.add_column("Scenario Category", style="cyan")
    summary_table.add_column("Resolution & FPS", justify="center")
    summary_table.add_column("Frames / Dur", justify="center")
    summary_table.add_column("File Size", justify="right")
    summary_table.add_column("Status", justify="center")

    success_count = 0
    for fname in selected:
        info = TEST_VIDEOS[fname]
        dest = out_dir / fname
        console.print(f"Fetching [bold]{fname}[/bold] ({info['description']})...")
        ok = download_video_with_progress(info["url"], dest)

        if ok:
            meta = verify_video(dest)
            if meta["valid"]:
                success_count += 1
                res_fps = f"{meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps"
                dur = f"{meta['frame_count']} f ({meta['duration_sec']:.1f}s)"
                size = f"{meta['size_mb']:.2f} MB"
                summary_table.add_row(fname, info["category"], res_fps, dur, size, "[bold green]VERIFIED[/bold green]")
            else:
                summary_table.add_row(fname, info["category"], "N/A", "N/A", "N/A", f"[bold red]{meta['error']}[/bold red]")
        else:
            summary_table.add_row(fname, info["category"], "N/A", "N/A", "N/A", "[bold red]DOWNLOAD FAILED[/bold red]")

    console.print()
    console.print(summary_table)

    if success_count == len(selected):
        console.print(f"\n[bold green]All {success_count} benchmark videos prepared and verified successfully![/bold green]")
        return 0
    else:
        console.print(f"\n[bold yellow]Completed with warnings: {success_count}/{len(selected)} verified.[/bold yellow]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
