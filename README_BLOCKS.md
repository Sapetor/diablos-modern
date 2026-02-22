# DiaBloS Blocks System - Quick Start Guide

This directory contains comprehensive documentation of the DiaBloS block system. Start here if you're new to DiaBloS or need to add/modify blocks.

## Documentation Files

### 1. **BLOCKS_ANALYSIS.md** (630 lines)
**Comprehensive technical analysis** - Read this for deep understanding.

**Contents**:
- Block inventory by category (76 blocks across 9 categories)
- BaseBlock class interface (required and optional properties)
- Parameter definition patterns and reusable templates
- Port definitions and types
- Block registration & discovery mechanism
- Complex block examples (PDE, Optimization, Integrator, Transfer Function)
- Architectural patterns (5 key patterns explained)
- Consistency analysis (strengths + minor issues)
- Key design decisions (trade-offs explained)
- Adding new blocks (step-by-step)
- Testing patterns
- System health summary

**Best for**: Understanding the "why" behind design choices, learning complex patterns, auditing consistency.

---

### 2. **BLOCKS_STRUCTURE.txt** (319 lines)
**Visual reference card** - Keep this handy while developing.

**Contents**:
- Inventory summary and distribution pie
- Directory structure with annotations
- Base block interface quick reference
- Parameter templates at a glance
- Registration mechanism explanation
- Design patterns enumerated (1-5)
- Consistency analysis summary
- Key design decisions with trade-offs
- Adding new blocks checklist
- Testing organization
- Health summary table
- Recommendations (high/medium/low priority)

**Best for**: Quick lookups, printing, sharing with team, reference while coding.

---

### 3. **BLOCKS_CHECKLIST.md** (434 lines)
**Development workflow checklist** - Use this when creating blocks.

**Contents**:
- Pre-creation planning checklist
- Block class implementation checklist
- Parameter definition requirements
- Port definition requirements
- Execute method signature and patterns
- Optional features (icons, symbolic execution, optional ports)
- Parameter template reference guide
- Code quality checklist
- Testing checklist with examples
- Subdirectory package setup
- Documentation requirements
- Pre-commit checklist
- Common pitfalls to avoid
- Post-creation verification
- Quick templates (minimal block, stateful block)
- Help resources

**Best for**: Step-by-step guidance while building new blocks, ensuring code quality, avoiding common mistakes.

---

## Quick Navigation

### I want to...

**Understand the block system**
- Read: BLOCKS_ANALYSIS.md (sections 1-5)
- Reference: BLOCKS_STRUCTURE.txt

**Add a new block**
- Use: BLOCKS_CHECKLIST.md (follow in order)
- Reference: BLOCKS_STRUCTURE.txt (for templates)
- Study: Similar existing blocks in `/blocks/`

**Debug a block**
- Check: BLOCKS_ANALYSIS.md (sections 7-9)
- Use: BLOCKS_CHECKLIST.md (code quality section)
- Review: Unit tests in `/tests/unit/`

**Maintain consistency**
- Read: BLOCKS_ANALYSIS.md (section 8)
- Reference: BLOCKS_STRUCTURE.txt (patterns section)
- Verify: BLOCKS_CHECKLIST.md (code quality)

**Extend the system**
- Study: BLOCKS_ANALYSIS.md (sections 9-10)
- Review: BLOCKS_STRUCTURE.txt (design decisions)
- Plan: BLOCKS_CHECKLIST.md (subdirectory setup)

---

## Key Statistics

**Block System**:
- 76 active blocks
- 9 categories
- 3 subdirectory packages
- 675 passing tests
- 57% unit test coverage

**Code Organization**:
- 47 top-level blocks (simple, math, control, routing)
- 11 PDE blocks (physics simulation)
- 11 Optimization Primitives (visual algorithm building)
- 5 Optimization blocks (parameter tuning)
- 1 base class (abstract)
- 1 inheritance base (StateSpaceBaseBlock)
- 2 utility modules (param_templates, input_helpers)

**Quality**:
- Architecture: Excellent
- Extensibility: Excellent
- Code reuse: Excellent
- Documentation: Good
- Consistency: Good
- Test coverage: Good
- Performance: Good

---

## Key Patterns

### 1. Parameter Templates (Reduce Duplication)
```python
@property
def params(self):
    return {
        **init_conds_param(default=0.0),
        **method_param(["RK45", "Euler"], "RK45"),
        **init_flag_param(),
        "custom_param": {...}
    }
```

### 2. Input Helpers (Safe Extraction)
```python
from blocks.input_helpers import get_scalar, get_vector, InitStateManager

def execute(self, time, inputs, params, **kwargs):
    init_mgr = InitStateManager(params)
    if init_mgr.needs_init():
        params['state'] = np.zeros(10)
        init_mgr.mark_initialized()
    
    u = get_scalar(inputs, 0)
    v = get_vector(inputs, 1, expected_dim=10)
    # ... rest of logic
```

### 3. Reflection-Based Registration (Auto-Discovery)
- No manual registry needed
- Just create a block file and class
- `load_blocks()` finds it automatically
- Works for top-level and subdirectory packages

### 4. Optional Ports (Flexible Connections)
```python
class MyBlock(BaseBlock):
    optional_inputs = {0}      # Port 0 doesn't need connection
    optional_outputs = {1}     # Port 1 doesn't need connection
```

### 5. Feedthrough Blocks (Immediate Response)
```python
@property
def b_type(self):
    return 2  # Output depends on input at same timestep
```

---

## File Locations

**Core System**:
- `/blocks/base_block.py` - Abstract base class
- `/blocks/param_templates.py` - Reusable parameter factories
- `/blocks/input_helpers.py` - Input extraction utilities
- `/lib/block_loader.py` - Reflection-based discovery

**Block Categories**:
- `/blocks/*.py` - Top-level blocks (47 files)
- `/blocks/pde/` - PDE and field processing blocks (9 files)
- `/blocks/optimization_primitives/` - Visual algorithm blocks (11 files)
- `/blocks/optimization/` - Parameter optimization blocks (5 files)

**Tests**:
- `/tests/unit/` - Block unit tests
- `/tests/integration/` - Full diagram tests
- `/tests/regression/` - Bug fix verification
- `/tests/profiling/` - Performance analysis

---

## Common Tasks

### Add a Simple Math Block
1. Open BLOCKS_CHECKLIST.md → "Quick Templates" → "Minimal Block"
2. Copy the template to `/blocks/myblock.py`
3. Implement the 8 required properties
4. Run: `python main.py` and verify it appears in palette
5. Write unit tests
6. Commit

**Estimated time**: 30 minutes

### Add a PDE Block
1. Study existing PDE block (e.g., HeatEquation1D)
2. Review BLOCKS_ANALYSIS.md section 6.1
3. Create `/blocks/pde/myblock.py`
4. Update `/blocks/pde/__init__.py`
5. Implement discretization logic
6. Write comprehensive tests
7. Add wiki page in `/docs/wiki/`
8. Commit

**Estimated time**: 4-8 hours

### Debug Parameter Issues
1. Check block's `params` property in BLOCKS_CHECKLIST.md
2. Verify parameter definition has `type`, `default`, `doc`
3. Use BLOCKS_ANALYSIS.md section 3 for parameter patterns
4. Check UI validation in modern_ui/
5. Run unit tests to isolate issue

**Estimated time**: 15-60 minutes

---

## Design Philosophy

The DiaBloS block system prioritizes:

1. **Simplicity** - Minimal required interface, composition over inheritance
2. **Extensibility** - Easy to add blocks without modifying core
3. **Consistency** - Uniform patterns across all blocks
4. **Documentation** - Every block and parameter documented
5. **Testing** - Comprehensive unit and integration tests
6. **Reusability** - Parameter templates and input helpers reduce duplication

---

## Recommendations

**Immediate**:
- Keep current design (working well)
- Add blocks as needed

**Soon**:
- Document StateSpaceBaseBlock inheritance pattern
- Add centralized color scheme constants
- Create block development guide (you're reading it!)

**Later**:
- Formal parameter schema validation
- Extend symbolic execution to more blocks
- Add block version metadata

---

## Help & Resources

**For implementation help**:
- Base class: `/blocks/base_block.py` (full interface documented)
- Examples: `/blocks/*.py` (review 3-4 similar blocks)
- Templates: `/blocks/param_templates.py` (factory functions)
- Helpers: `/blocks/input_helpers.py` (input extraction)

**For understanding**:
- BLOCKS_ANALYSIS.md (comprehensive reference)
- BLOCKS_STRUCTURE.txt (visual reference)
- Wiki pages: `/docs/wiki/` (complex examples)

**For testing**:
- Test examples: `/tests/unit/test_*.py`
- Patterns: BLOCKS_CHECKLIST.md (testing section)

**For design questions**:
- Review BLOCKS_ANALYSIS.md (sections 7-9)
- Look at similar block implementations
- Check project CLAUDE.md for principles

---

## Summary

This is a well-designed, maintainable block system with:
- 76 blocks across 9 categories
- Clear abstraction layer (BaseBlock)
- Reusable parameter templates
- Safe input extraction helpers
- Automatic block discovery
- 675 tests (57% coverage)
- Comprehensive documentation

Start with the appropriate document based on your task, follow the checklists, and you'll be adding blocks confidently in no time.

---

**Last Updated**: February 5, 2026
**Block Count**: 76 active blocks
**Test Count**: 675 passing tests
**System Status**: Excellent health
