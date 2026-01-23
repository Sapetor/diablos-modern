import os
import sys
import importlib.util
import inspect
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_block_classes(blocks_dir):
    """Scan blocks directory and load all block classes."""
    blocks = []
    
    for string_path in os.listdir(blocks_dir):
        if not string_path.endswith(".py") or string_path.startswith("__") or string_path == "base_block.py":
            continue
            
        module_name = string_path[:-3]
        file_path = blocks_dir / string_path
        
        try:
            spec = importlib.util.spec_from_file_location(f"blocks.{module_name}", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the block class in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, 'block_name') and hasattr(obj, 'category'):
                    # Instantiate to get properties
                    try:
                        instance = obj()
                        blocks.append(instance)
                        break # Only one block per file usually
                    except Exception as e:
                        print(f"Skipping {name}: Could not instantiate ({e})")
                        
        except Exception as e:
            print(f"Error loading {string_path}: {e}")
            
    return blocks

def ensure_wiki_dir(docs_dir):
    """Ensure docs/wiki directory exists."""
    wiki_dir = docs_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    return wiki_dir

def generate_sidebar(categories):
    """Generate _Sidebar.md content."""
    lines = ["## Navigation", ""]
    lines.append("- [Home](Home)")
    lines.append("")
    lines.append("### Categories")
    
    for category in sorted(categories):
        link_name = category.replace(' ', '-')
        lines.append(f"- [{category}]({link_name})")
        
    return "\n".join(lines)

def generate_home(categories):
    """Generate Home.md content."""
    lines = [
        "# Welcome to Modern DiaBloS Wiki",
        "",
        "Welcome to the official documentation for **Modern DiaBloS**.",
        "",
        "## Block Library",
        "Browse blocks by category:",
        ""
    ]
    
    for category in sorted(categories):
        link_name = category.replace(' ', '-')
        lines.append(f"- **[{category}]({link_name})**")
        
    lines.append("")
    lines.append("## Core Documentation")
    lines.append("- [User Manual](../USER_MANUAL.md)")
    lines.append("- [Developer Guide](../DEVELOPER_GUIDE.md)")
    
    return "\n".join(lines)

def generate_category_page(category, blocks):
    """Generate contents for a category page (e.g. Control.md)."""
    lines = [
        f"# {category} Blocks",
        "",
        f"List of available blocks in the **{category}** category.",
        "",
        "You can find detailed information about parameters and usage below.",
        "",
        "| Block | Description |",
        "|-------|-------------|",
    ]
    
    # Sort blocks by name
    blocks.sort(key=lambda b: b.block_name)
    
    for block in blocks:
        short_desc = block.doc.split('\n')[0] if block.doc else "No description"
        clean_name = block.block_name.lower().replace(' ', '-')
        lines.append(f"| [{block.block_name}](#{clean_name}) | {short_desc} |")
        
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed sections
    for block in blocks:
        lines.append(f"### {block.block_name}")
        lines.append("")
        
        # Doc
        lines.append(block.doc if block.doc else "_No documentation available._")
        lines.append("")
        
        # Params
        if hasattr(block, 'params') and block.params:
            lines.append(f"#### Parameters")
            lines.append(f"| Name | Type | Default | Description |")
            lines.append(f"|------|------|---------|-------------|")
            
            for param, info in block.params.items():
                if param.startswith('_'): continue
                
                p_type = info.get('type', 'any')
                p_default = str(info.get('default', ''))
                p_doc = info.get('doc', '')
                
                lines.append(f"| `{param}` | {p_type} | `{p_default}` | {p_doc} |")
            
            lines.append("")
        
        # Ports
        inputs = getattr(block, 'inputs', [])
        outputs = getattr(block, 'outputs', [])
        if inputs or outputs:
            in_count = len(inputs) if isinstance(inputs, list) else "Dynamic"
            out_count = len(outputs) if isinstance(outputs, list) else "Dynamic"
            lines.append(f"**Ports**: {in_count} In, {out_count} Out")
            
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)

def main():
    print("Initializing Wiki Generator...")
    blocks_dir = PROJECT_ROOT / "blocks"
    docs_dir = PROJECT_ROOT / "docs"
    wiki_dir = ensure_wiki_dir(docs_dir)
    
    # Load blocks
    blocks = load_block_classes(blocks_dir)
    print(f"Loaded {len(blocks)} blocks.")
    
    # Group by category
    categories = {}
    for block in blocks:
        cat = block.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(block)
        
    # 1. Generate Sidebar
    sidebar_content = generate_sidebar(categories.keys())
    with open(wiki_dir / "_Sidebar.md", 'w', encoding='utf-8') as f:
        f.write(sidebar_content)
    print("Generated _Sidebar.md")
    
    # 2. Generate Home
    home_content = generate_home(categories.keys())
    with open(wiki_dir / "Home.md", 'w', encoding='utf-8') as f:
        f.write(home_content)
    print("Generated Home.md")
    
    # 3. Generate Category Pages
    for category, category_blocks in categories.items():
        filename = f"{category.replace(' ', '-')}.md"
        content = generate_category_page(category, category_blocks)
        
        with open(wiki_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Generated {filename}")
        
    print(f"Wiki generation complete! Output: {wiki_dir}")

if __name__ == "__main__":
    main()
