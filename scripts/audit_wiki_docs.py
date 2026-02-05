#!/usr/bin/env python3
"""
Wiki Documentation Audit Script

Scans all block files for block_name and category properties,
parses wiki files for anchors, and reports:
- Missing wiki files for categories
- Missing anchors for blocks
- Category mismatches

Usage:
    python scripts/audit_wiki_docs.py
"""

import os
import re
import sys
from pathlib import Path


def find_project_root():
    """Find the project root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def scan_block_files(blocks_dir):
    """
    Scan all block files and extract block_name and category.

    Returns:
        List of dicts with 'file', 'block_name', 'category'
    """
    blocks = []

    for root, dirs, files in os.walk(blocks_dir):
        # Skip __pycache__ and test directories
        dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('test')]

        for filename in files:
            if not filename.endswith('.py'):
                continue
            if filename.startswith('__') or filename == 'base_block.py':
                continue

            filepath = Path(root) / filename
            try:
                content = filepath.read_text()

                # Extract block_name property
                block_name_match = re.search(
                    r'def block_name\(self\):\s*\n\s*return\s*["\']([^"\']+)["\']',
                    content
                )

                # Extract category property
                category_match = re.search(
                    r'def category\(self\):\s*\n\s*return\s*["\']([^"\']+)["\']',
                    content
                )

                if block_name_match:
                    block_info = {
                        'file': str(filepath.relative_to(blocks_dir.parent)),
                        'block_name': block_name_match.group(1),
                        'category': category_match.group(1) if category_match else 'Unknown'
                    }
                    blocks.append(block_info)

            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")

    return blocks


def parse_wiki_files(wiki_dir):
    """
    Parse wiki files and extract anchors (### headers).

    Returns:
        Dict mapping filename (without .md) to list of anchors
    """
    wiki_anchors = {}

    for filepath in wiki_dir.glob('*.md'):
        if filepath.name.startswith('_'):
            continue

        filename = filepath.stem  # filename without .md
        content = filepath.read_text()

        # Find all ## and ### headers (block documentation sections)
        # GitHub creates anchors from any heading level
        anchors = re.findall(r'^#{2,3}\s+(\S+)', content, re.MULTILINE)
        # Normalize anchors to lowercase with hyphens (GitHub anchor format)
        normalized_anchors = [a.lower().replace(' ', '-') for a in anchors]

        wiki_anchors[filename] = {
            'raw_anchors': anchors,
            'normalized': normalized_anchors
        }

    return wiki_anchors


def compute_expected_wiki_file(category):
    """
    Compute the expected wiki filename from a category.
    Mirrors the logic in property_editor.py.
    """
    return category.replace(' ', '-')


def audit_documentation(blocks, wiki_anchors):
    """
    Audit blocks against wiki documentation.

    Returns:
        Dict with 'missing_wiki_files', 'missing_anchors', 'category_mismatches'
    """
    issues = {
        'missing_wiki_files': [],
        'missing_anchors': [],
        'working_links': []
    }

    available_wiki_files = set(wiki_anchors.keys())

    for block in blocks:
        block_name = block['block_name']
        category = block['category']
        expected_wiki_file = compute_expected_wiki_file(category)
        expected_anchor = block_name.lower().replace(' ', '-')

        # Check if wiki file exists
        if expected_wiki_file not in available_wiki_files:
            # Check for underscore variant
            underscore_variant = expected_wiki_file.replace('-', '_')
            if underscore_variant in available_wiki_files:
                issues['missing_wiki_files'].append({
                    'block': block_name,
                    'category': category,
                    'expected_file': f"{expected_wiki_file}.md",
                    'actual_file': f"{underscore_variant}.md",
                    'file': block['file'],
                    'issue': 'underscore_vs_hyphen'
                })
            else:
                issues['missing_wiki_files'].append({
                    'block': block_name,
                    'category': category,
                    'expected_file': f"{expected_wiki_file}.md",
                    'file': block['file'],
                    'issue': 'file_not_found'
                })
            continue

        # Check if anchor exists in wiki file
        wiki_data = wiki_anchors[expected_wiki_file]
        if expected_anchor not in wiki_data['normalized']:
            issues['missing_anchors'].append({
                'block': block_name,
                'category': category,
                'wiki_file': f"{expected_wiki_file}.md",
                'expected_anchor': expected_anchor,
                'available_anchors': wiki_data['raw_anchors'],
                'file': block['file']
            })
        else:
            issues['working_links'].append({
                'block': block_name,
                'category': category,
                'url': f"{expected_wiki_file}.md#{expected_anchor}"
            })

    return issues


def print_report(issues):
    """Print a formatted audit report."""
    print("=" * 70)
    print("WIKI DOCUMENTATION AUDIT REPORT")
    print("=" * 70)

    # Missing wiki files
    print(f"\n## Missing Wiki Files ({len(issues['missing_wiki_files'])} issues)\n")
    if issues['missing_wiki_files']:
        for item in issues['missing_wiki_files']:
            if item['issue'] == 'underscore_vs_hyphen':
                print(f"- **{item['block']}** (category: `{item['category']}`)")
                print(f"  - Expected: `{item['expected_file']}`")
                print(f"  - Found: `{item['actual_file']}` (underscore vs hyphen mismatch)")
                print(f"  - File: `{item['file']}`")
            else:
                print(f"- **{item['block']}** (category: `{item['category']}`)")
                print(f"  - Expected: `{item['expected_file']}` (NOT FOUND)")
                print(f"  - File: `{item['file']}`")
    else:
        print("No missing wiki files!")

    # Missing anchors
    print(f"\n## Missing Anchors ({len(issues['missing_anchors'])} issues)\n")
    if issues['missing_anchors']:
        for item in issues['missing_anchors']:
            print(f"- **{item['block']}** in `{item['wiki_file']}`")
            print(f"  - Expected anchor: `#{item['expected_anchor']}`")
            print(f"  - Available: {item['available_anchors'][:5]}...")
            print(f"  - File: `{item['file']}`")
    else:
        print("No missing anchors!")

    # Working links summary
    print(f"\n## Working Links ({len(issues['working_links'])} blocks)\n")
    if issues['working_links']:
        by_category = {}
        for item in issues['working_links']:
            cat = item['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item['block'])
        for cat, blocks in sorted(by_category.items()):
            print(f"- **{cat}**: {', '.join(blocks)}")

    # Summary
    total_issues = len(issues['missing_wiki_files']) + len(issues['missing_anchors'])
    print(f"\n## Summary\n")
    print(f"- Total blocks scanned: {len(issues['working_links']) + total_issues}")
    print(f"- Working links: {len(issues['working_links'])}")
    print(f"- Issues found: {total_issues}")

    return total_issues


def main():
    project_root = find_project_root()
    blocks_dir = project_root / 'blocks'
    wiki_dir = project_root / 'docs' / 'wiki'

    print(f"Project root: {project_root}")
    print(f"Scanning blocks in: {blocks_dir}")
    print(f"Scanning wiki in: {wiki_dir}")

    # Scan blocks
    blocks = scan_block_files(blocks_dir)
    print(f"Found {len(blocks)} blocks with block_name property")

    # Parse wiki files
    wiki_anchors = parse_wiki_files(wiki_dir)
    print(f"Found {len(wiki_anchors)} wiki files")

    # Audit
    issues = audit_documentation(blocks, wiki_anchors)

    # Print report
    total_issues = print_report(issues)

    # Exit code
    sys.exit(0 if total_issues == 0 else 1)


if __name__ == '__main__':
    main()
