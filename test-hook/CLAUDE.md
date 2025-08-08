# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python educational repository demonstrating the differences between **hooks** and **callbacks** programming patterns. The codebase contains 6 example files that illustrate various aspects of these patterns.

## Commands

Since this is a simple Python example repository without package.json or build configuration:

- **Run examples**: `python <filename>.py` (e.g., `python hook_example.py`)
- **Run all examples in sequence**: 
  ```bash
  python callback_example.py && python hook_example.py && python practical_example.py && python simple_timing.py && python timing_example.py && python summary.py
  ```
- **Python linting** (if available): `python -m flake8 .` or `python -m pylint *.py`
- **Type checking** (if available): `python -m mypy *.py`

## Code Architecture

The repository demonstrates two fundamental programming patterns through practical examples:

### Core Pattern Files
- **callback_example.py**: Basic callback pattern - functions passed as arguments to be called later
- **hook_example.py**: Event-driven hook system using EventSystem class that manages hook registration and triggering
- **practical_example.py**: Real-world example combining both patterns in a FileProcessor class with before/after hooks and optional callbacks

### Educational Files
- **simple_timing.py**: Demonstrates the key timing difference between hooks (fixed timing when event triggers) and callbacks (flexible timing controlled by receiver)
- **timing_example.py**: Advanced timing examples with HookSystem class and time delays to show execution flow
- **summary.py**: Contains documentation strings and explanations of the key differences between patterns

### Key Architectural Concepts

**Hook Pattern** (`hook_example.py:1-16`):
- Uses an EventSystem class to manage hooks
- Multiple functions can be registered for the same event
- Fixed timing - all hooks execute when event is triggered
- Event-driven architecture

**Callback Pattern** (`callback_example.py:1-4`):
- Functions passed as parameters to other functions  
- Flexible timing - receiving function controls when callback is called
- Typically one-to-one relationship between caller and callback

**Combined Usage** (`practical_example.py:14-32`):
- FileProcessor class uses before/after hook arrays
- Optional callback parameter for final result processing
- Demonstrates how both patterns complement each other in real applications

The examples progress from basic concepts to practical implementations, making this repository ideal for learning the differences between these two important programming patterns.