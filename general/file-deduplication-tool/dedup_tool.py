"""
File Deduplication Tool
=======================
Finds duplicate files by SHA256 hash, displays results
in Rich tables, and calculates reclaimable disk space.

Usage:
    python dedup_tool.py                     # scan built-in test directory
    python dedup_tool.py /path/to/directory   # scan a real directory

Requirements:
    pip install rich
"""
import hashlib
import os
import shutil
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.panel import Panel

console = Console()

# ═══════════════════════════════════════════════════════════════
# HASHING
# ═══════════════════════════════════════════════════════════════

def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file efficiently using chunked reading."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (PermissionError, OSError) as e:
        return f"ERROR:{e}"

# ═══════════════════════════════════════════════════════════════
# SCANNER: TWO-PASS DEDUPLICATION
# ═══════════════════════════════════════════════════════════════

def scan_directory(root_dir: Path, min_size: int = 1) -> Tuple[Dict[str, List[Path]], int, int]:
    """
    Scan directory and group files by SHA256 hash.

    First pass:  group by file size (fast pre-filter).
    Second pass: hash only files that share a size with another file.

    Returns:
        (hash->files mapping, total_files, total_size)
    """
    size_groups: Dict[int, List[Path]] = defaultdict(list)
    total_files = 0
    total_size = 0

    # First pass: group by file size
    for filepath in root_dir.rglob("*"):
        if filepath.is_file() and not filepath.is_symlink():
            try:
                fsize = filepath.stat().st_size
                if fsize >= min_size:
                    size_groups[fsize].append(filepath)
                    total_files += 1
                    total_size += fsize
            except OSError:
                continue

    # Second pass: hash files with size collisions
    hash_map: Dict[str, List[Path]] = defaultdict(list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        files_to_hash = sum(
            len(files) for files in size_groups.values() if len(files) > 1
        )

        if files_to_hash == 0:
            return hash_map, total_files, total_size

        task = progress.add_task("[cyan]Hashing files...", total=files_to_hash)

        for fsize, files in size_groups.items():
            if len(files) > 1:
                for filepath in files:
                    file_hash = get_file_hash(filepath)
                    hash_map[file_hash].append(filepath)
                    progress.advance(task)

    return hash_map, total_files, total_size


def find_duplicates(hash_map: Dict[str, List[Path]]) -> List[Tuple[str, List[Path]]]:
    """Filter to only entries where 2+ files share the same hash."""
    return [(h, files) for h, files in hash_map.items() if len(files) > 1]

# ═══════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def display_results(
    duplicates: List[Tuple[str, List[Path]]],
    total_files: int,
    total_size: int,
    root_dir: Path
):
    """Display duplicate files in Rich tables with summary stats."""
    if not duplicates:
        console.print(Panel(
            f"[green]No duplicate files found in [bold]{root_dir}[/bold]![/green]",
            title="Scan Complete"
        ))
        return

    # Calculate wasted space
    wasted_files = sum(len(files) - 1 for _, files in duplicates)
    wasted_bytes = 0
    for _, files in duplicates:
        file_size = files[0].stat().st_size
        wasted_bytes += file_size * (len(files) - 1)

    # Summary panel
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column(style="white")
    summary.add_row("Directory scanned:", str(root_dir))
    summary.add_row("Total files:", f"{total_files:,}")
    summary.add_row("Total size:", format_size(total_size))
    summary.add_row("Duplicate groups:", f"[yellow]{len(duplicates)}[/yellow]")
    summary.add_row("Wasted files:", f"[red]{wasted_files}[/red]")
    summary.add_row("Wasted space:", f"[red bold]{format_size(wasted_bytes)}[/red bold]")

    console.print(Panel(summary, title="Scan Summary", border_style="blue"))

    # Duplicate groups table
    table = Table(title="Duplicate Files Found", show_lines=True)
    table.add_column("Group", style="cyan", width=6)
    table.add_column("File Path", style="white")
    table.add_column("Size", style="yellow", width=12)
    table.add_column("Status", width=10)

    for i, (file_hash, files) in enumerate(duplicates, 1):
        file_size = format_size(files[0].stat().st_size)
        for j, fpath in enumerate(files):
            rel_path = str(fpath.relative_to(root_dir))
            status = "[green]KEEP[/green]" if j == 0 else "[red]DUPLICATE[/red]"
            table.add_row(
                str(i) if j == 0 else "",
                rel_path,
                file_size if j == 0 else "",
                status
            )

    console.print(table)

    # Recommendation
    console.print(Panel(
        f"[yellow]To reclaim [bold]{format_size(wasted_bytes)}[/bold], review the "
        f"[red]DUPLICATE[/red] files above and delete the copies you don't need. "
        f"Keep one copy in each group ([green]KEEP[/green]).[/yellow]",
        title="Recommendation",
        border_style="yellow"
    ))

# ═══════════════════════════════════════════════════════════════
# TEST SETUP (for demonstration)
# ═══════════════════════════════════════════════════════════════

def setup_test_files(base_dir: str = "test_files"):
    """Create a directory structure with deliberate duplicates for testing."""
    if Path(base_dir).exists():
        shutil.rmtree(base_dir)
    Path(base_dir).mkdir(exist_ok=True)

    dirs = ["photos", "documents", "downloads", "photos/vacation", "documents/old"]
    for d in dirs:
        Path(base_dir, d).mkdir(parents=True, exist_ok=True)

    random.seed(42)

    file_records = []
    for i in range(30):
        size = random.choice([1024, 5120, 10240, 51200, 102400])
        content = os.urandom(size)
        folder = random.choice(dirs)
        ext = random.choice([".txt", ".jpg", ".png", ".pdf", ".docx", ".csv"])
        name = f"file_{i:03d}{ext}"
        file_records.append((folder, name, content))

    # Plant duplicates (same content, different names/locations)
    duplicate_plan = [
        (0, "photos/vacation", "beach_photo.jpg"),
        (0, "downloads", "temp_image.jpg"),        # triplicate!
        (5, "documents/old", "old_report.pdf"),
        (10, "downloads", "budget_backup.csv"),
        (15, "photos", "profile_pic_copy.png"),
        (20, "documents/old", "archived_notes.docx"),
    ]

    for orig_idx, dup_folder, dup_name in duplicate_plan:
        folder, name, content = file_records[orig_idx]
        Path(base_dir, dup_folder).mkdir(parents=True, exist_ok=True)
        Path(base_dir, dup_folder, dup_name).write_bytes(content)

    for folder, name, content in file_records:
        Path(base_dir, folder, name).write_bytes(content)

    return base_dir

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point."""

    # Accept a directory path from the command line, or use test files
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        console.print(f"[bold]Scanning [cyan]{target_dir}[/cyan] for duplicates...[/bold]\n")
    else:
        console.print("[bold]Setting up test files...[/bold]")
        target_dir = setup_test_files("test_files")
        console.print("[bold]Scanning for duplicates...[/bold]\n")

    hash_map, total_files, total_size = scan_directory(Path(target_dir))
    duplicates = find_duplicates(hash_map)
    display_results(duplicates, total_files, total_size, Path(target_dir))


if __name__ == "__main__":
    main()
