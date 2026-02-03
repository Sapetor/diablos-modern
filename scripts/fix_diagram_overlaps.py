#!/usr/bin/env python3
"""
Fix line-block overlaps in DiaBloS diagram files.

This script detects when line segments pass through blocks and reroutes them
by adding waypoints that go around the blocks with a configurable margin.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Margin around blocks when routing (pixels)
MARGIN = 15


def load_diagram(filepath: Path) -> Dict[str, Any]:
    """Load a DiaBloS diagram JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_diagram(filepath: Path, data: Dict[str, Any]) -> None:
    """Save a DiaBloS diagram JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def get_block_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Get block bounding box with margin.

    Returns: (left, right, top, bottom)
    """
    left = block['coords_left'] - MARGIN
    right = block['coords_left'] + block['coords_width'] + MARGIN
    top = block['coords_top'] - MARGIN
    bottom = block['coords_top'] + block['coords_height'] + MARGIN
    return left, right, top, bottom


def point_in_bbox(p: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
    """Check if point is inside bounding box."""
    left, right, top, bottom = bbox
    x, y = p
    return left < x < right and top < y < bottom


def segment_intersects_bbox(p1: Tuple[float, float], p2: Tuple[float, float],
                            bbox: Tuple[float, float, float, float]) -> bool:
    """
    Check if line segment from p1 to p2 intersects the bounding box.

    Args:
        p1: (x, y) start point
        p2: (x, y) end point
        bbox: (left, right, top, bottom)

    Returns:
        True if segment intersects bbox interior
    """
    left, right, top, bottom = bbox
    x1, y1 = p1
    x2, y2 = p2

    # Check if both endpoints are outside on the same side
    if x1 < left and x2 < left:
        return False
    if x1 > right and x2 > right:
        return False
    if y1 < top and y2 < top:
        return False
    if y1 > bottom and y2 > bottom:
        return False

    # Check if either endpoint is inside the bbox
    if point_in_bbox(p1, bbox) or point_in_bbox(p2, bbox):
        return True

    # Check if segment crosses the bbox edges
    dx = x2 - x1
    dy = y2 - y1

    if dx != 0:
        # Check left edge
        t = (left - x1) / dx
        if 0 < t < 1:
            y = y1 + t * dy
            if top < y < bottom:
                return True

        # Check right edge
        t = (right - x1) / dx
        if 0 < t < 1:
            y = y1 + t * dy
            if top < y < bottom:
                return True

    if dy != 0:
        # Check top edge
        t = (top - y1) / dy
        if 0 < t < 1:
            x = x1 + t * dx
            if left < x < right:
                return True

        # Check bottom edge
        t = (bottom - y1) / dy
        if 0 < t < 1:
            x = x1 + t * dx
            if left < x < right:
                return True

    return False


def find_overlapping_blocks(line: Dict[str, Any], blocks: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Find all blocks that the line segments pass through.

    Returns: List of (segment_index, block) tuples
    """
    overlaps = []
    points = line['points']

    for i in range(len(points) - 1):
        p1 = tuple(points[i])
        p2 = tuple(points[i + 1])

        for block in blocks:
            # Skip blocks that are the source or destination
            if block['username'] == line['srcblock'] or block['username'] == line['dstblock']:
                continue

            bbox = get_block_bbox(block)
            if segment_intersects_bbox(p1, p2, bbox):
                overlaps.append((i, block))

    return overlaps


def route_around_block(p1: Tuple[float, float], p2: Tuple[float, float],
                       block: Dict[str, Any], all_blocks: List[Dict[str, Any]],
                       src_block: str, dst_block: str) -> List[Tuple[float, float]]:
    """
    Create waypoints to route around a block using a simple rectangular path.
    Tries multiple routing options and picks one that doesn't create new overlaps.

    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        block: Block dictionary to route around
        all_blocks: All blocks in diagram for collision checking
        src_block: Source block name (to skip in collision check)
        dst_block: Destination block name (to skip in collision check)

    Returns:
        List of waypoints including p1 and p2
    """
    left, right, top, bottom = get_block_bbox(block)
    x1, y1 = p1
    x2, y2 = p2

    # Generate candidate routes (try all 4 directions)
    candidates = [
        # Route above
        [p1, (x1, top - 5), (x2, top - 5), p2],
        # Route below
        [p1, (x1, bottom + 5), (x2, bottom + 5), p2],
        # Route left
        [p1, (left - 5, y1), (left - 5, y2), p2],
        # Route right
        [p1, (right + 5, y1), (right + 5, y2), p2],
    ]

    def count_overlaps(route: List[Tuple[float, float]]) -> int:
        """Count how many blocks a route overlaps."""
        count = 0
        for i in range(len(route) - 1):
            seg_p1 = route[i]
            seg_p2 = route[i + 1]
            for b in all_blocks:
                if b['username'] == src_block or b['username'] == dst_block:
                    continue
                bbox = get_block_bbox(b)
                if segment_intersects_bbox(seg_p1, seg_p2, bbox):
                    count += 1
        return count

    def route_length(route: List[Tuple[float, float]]) -> float:
        """Calculate total route length."""
        total = 0.0
        for i in range(len(route) - 1):
            dx = route[i+1][0] - route[i][0]
            dy = route[i+1][1] - route[i][1]
            total += (dx*dx + dy*dy) ** 0.5
        return total

    # Score each candidate: prefer fewer overlaps, then shorter length
    best_route = candidates[0]
    best_score = (count_overlaps(best_route), route_length(best_route))

    for route in candidates[1:]:
        score = (count_overlaps(route), route_length(route))
        if score < best_score:
            best_score = score
            best_route = route

    return best_route


def fix_line(line: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """
    Fix a single line by routing around overlapping blocks.
    Only processes each segment once to avoid creating loops.

    Returns:
        (modified, new_line_dict)
    """
    overlaps = find_overlapping_blocks(line, blocks)

    if not overlaps:
        return False, line

    # Group overlaps by segment
    segments_to_fix = {}
    for seg_idx, block in overlaps:
        if seg_idx not in segments_to_fix:
            segments_to_fix[seg_idx] = []
        segments_to_fix[seg_idx].append(block)

    # Build new points list by processing each segment
    points = line['points']
    new_points = [points[0]]  # Start with first point

    for i in range(len(points) - 1):
        p1 = tuple(points[i])
        p2 = tuple(points[i + 1])

        if i in segments_to_fix:
            # This segment needs fixing - fix based on first overlapping block
            block = segments_to_fix[i][0]
            waypoints = route_around_block(p1, p2, block, blocks, line['srcblock'], line['dstblock'])

            # Add intermediate waypoints (skip first point as it's already in new_points)
            new_points.extend(waypoints[1:])
        else:
            # No overlap, just add the end point
            new_points.append(list(p2))

    # Create new line dict
    new_line = line.copy()
    new_line['points'] = new_points

    return True, new_line


def fix_diagram(filepath: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Fix all line-block overlaps in a diagram file.

    Returns:
        (total_overlaps_found, lines_modified)
    """
    data = load_diagram(filepath)
    blocks = data.get('blocks_data', [])
    lines = data.get('lines_data', [])

    total_overlaps = 0
    lines_modified = 0
    new_lines = []

    for line in lines:
        overlaps = find_overlapping_blocks(line, blocks)
        overlap_count = len(overlaps)
        total_overlaps += overlap_count

        if overlap_count > 0:
            if not dry_run:
                modified, new_line = fix_line(line, blocks)
                if modified:
                    lines_modified += 1
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                lines_modified += 1
                new_lines.append(line)
        else:
            new_lines.append(line)

    if not dry_run and lines_modified > 0:
        data['lines_data'] = new_lines
        save_diagram(filepath, data)

    return total_overlaps, lines_modified


def report_overlaps(filepath: Path) -> None:
    """Print detailed report of all overlaps in a diagram."""
    data = load_diagram(filepath)
    blocks = data.get('blocks_data', [])
    lines = data.get('lines_data', [])

    print(f"\n=== {filepath.name} ===")

    for line in lines:
        overlaps = find_overlapping_blocks(line, blocks)
        if overlaps:
            print(f"  Line {line['sid']}: {line['srcblock']} -> {line['dstblock']}")
            for seg_idx, block in overlaps:
                pts = line['points']
                p1, p2 = pts[seg_idx], pts[seg_idx + 1]
                print(f"    Seg {seg_idx} [{p1[0]},{p1[1]}]->[{p2[0]},{p2[1]}] overlaps '{block['username']}' " +
                      f"(x:{block['coords_left']}-{block['coords_left']+block['coords_width']}, " +
                      f"y:{block['coords_top']}-{block['coords_top']+block['coords_height']})")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix line-block overlaps in DiaBloS diagrams')
    parser.add_argument('files', nargs='*', help='Diagram files to fix (default: all in examples/)')
    parser.add_argument('--dry-run', action='store_true', help='Only report issues without fixing')
    parser.add_argument('--examples-dir', default='examples', help='Directory containing examples')
    parser.add_argument('--max-passes', type=int, default=5, help='Maximum fix passes per file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed overlap info')
    parser.add_argument('--report', action='store_true', help='Show detailed overlap report')

    args = parser.parse_args()

    # Determine which files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        # Process all .diablos files in examples directory
        examples_dir = Path(args.examples_dir)
        if not examples_dir.exists():
            print(f"Error: Directory '{examples_dir}' not found")
            return 1

        files = sorted(examples_dir.glob('*.diablos'))

    if not files:
        print("No .diablos files found")
        return 1

    # Report mode - show detailed overlap info
    if args.report:
        for filepath in files:
            try:
                data = load_diagram(filepath)
                blocks = data.get('blocks_data', [])
                lines = data.get('lines_data', [])
                total = sum(len(find_overlapping_blocks(line, blocks)) for line in lines)
                if total > 0:
                    report_overlaps(filepath)
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
        return 0

    # Process each file
    total_issues = 0
    total_fixed = 0

    print(f"{'Checking' if args.dry_run else 'Fixing'} {len(files)} diagram files...")
    print()

    for filepath in files:
        try:
            if args.dry_run:
                overlaps, modified = fix_diagram(filepath, dry_run=True)
                if overlaps > 0:
                    total_issues += overlaps
                    total_fixed += modified
                    print(f"  {filepath.name}: {overlaps} overlaps detected, would fix {modified} lines")
            else:
                # Multi-pass fixing
                initial_overlaps, _ = fix_diagram(filepath, dry_run=True)
                if initial_overlaps == 0:
                    continue

                for pass_num in range(args.max_passes):
                    overlaps, modified = fix_diagram(filepath, dry_run=False)
                    if modified == 0:
                        break

                # Check final state
                final_overlaps, _ = fix_diagram(filepath, dry_run=True)
                total_issues += initial_overlaps
                total_fixed += (initial_overlaps - final_overlaps)

                if args.verbose or final_overlaps > 0:
                    print(f"  {filepath.name}: {initial_overlaps} -> {final_overlaps} overlaps")
        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    print()
    print(f"Summary: {total_issues} overlaps in {total_fixed} lines")

    if args.dry_run:
        print("(Dry run - no files modified)")
    else:
        print(f"Fixed {total_fixed} lines across {len([f for f in files])} files")

    return 0


if __name__ == '__main__':
    sys.exit(main())
