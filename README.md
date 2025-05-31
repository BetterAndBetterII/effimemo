# EffiMemo

[![PyPI version](https://badge.fury.io/py/effimemo.svg)](https://badge.fury.io/py/effimemo)
[![Python Support](https://img.shields.io/pypi/pyversions/effimemo.svg)](https://pypi.org/project/effimemo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/BetterAndBetterII/effimemo)

A Python package for managing Large Language Model (LLM) context windows with intelligent compression and multiple truncation strategies.

[中文文档](README_CN.md) | English

## Features

- **Intelligent Context Management**: Automatically manage conversation history to ensure token limits are not exceeded
- **Multiple Compression Strategies**: Support for `first`, `last`, `selective`, and `summary` strategies
- **Flexible Configuration**: Customizable maximum tokens, model types, and other parameters
- **System Message Protection**: Optional preservation of important system messages
- **OpenAI Integration**: Support for OpenAI API for intelligent summary compression
- **Tool Calling Support**: Full support for OpenAI function calling and tool usage
- **High Test Coverage**: 80%+ test coverage ensuring code quality

## Installation

### Basic Installation
```bash
pip install effimemo
```

### Optional Dependencies
```bash
# Support for OpenAI summary strategy
pip install effimemo[openai]

# Support for selective compression strategy
pip install effimemo[compression]

# Development dependencies
pip install effimemo[dev]

# Install all dependencies
pip install effimemo[openai,compression,dev]
```

## Quick Start

### Basic Usage

```python
from effimemo import create_context_manager

# Create context manager
manager = create_context_manager(
    max_tokens=8192,
    model_name="gpt-4",
    strategy="last",
    preserve_system=True
)

# Use manager to process conversations
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "Tell me about quantum physics"}
]

# Compress context (reserve 1000 tokens for response)
compressed_messages = manager.compress(messages, reserve_tokens=1000)

# Count tokens
token_count = manager.count_tokens(messages)
print(f"Original message token count: {token_count}")
```

### Direct ContextManager Usage

```python
from effimemo import ContextManager

# Create manager instance
manager = ContextManager(
    max_tokens=4096,
    model_name="gpt-3.5-turbo",
    strategy="summary",
    preserve_system=True
)

# Process long conversations
long_conversation = [
    {"role": "system", "content": "You are a professional programming assistant"},
    # ... many conversation messages
]

# Compress conversation
result = manager.compress(long_conversation)
```

## Compression Strategies

### 1. Last Strategy (Default)
Keep recent messages, remove earlier messages:
```python
manager = create_context_manager(strategy="last")
```

### 2. First Strategy
Keep earliest messages, remove newer messages:
```python
manager = create_context_manager(strategy="first")
```

### 3. Selective Strategy
Use intelligent compression algorithms to reduce message content:
```python
# Requires: pip install effimemo[compression]
manager = create_context_manager(strategy="selective")
```

### 4. Summary Strategy
Use OpenAI API to generate conversation summaries:
```python
import openai

# Requires: pip install effimemo[openai]
client = openai.OpenAI(api_key="your-api-key")

manager = create_context_manager(
    strategy="summary",
    openai_client=client,
    summary_model="gpt-3.5-turbo",
    preserve_recent=3,  # Keep last 3 messages
    summary_prompt="Please concisely summarize the following conversation:\n{conversation}"
)
```

## Advanced Usage

### Custom Parameters

```python
from effimemo import ContextManager

manager = ContextManager(
    max_tokens=8192,
    model_name="gpt-4",
    strategy="summary",
    preserve_system=True,
    # Summary strategy parameters
    openai_client=openai_client,
    summary_model="gpt-4",
    preserve_recent=5,
    summary_prompt="Custom summary prompt: {conversation}",
    # Truncation strategy parameters
    min_content_tokens=50
)
```

### Message Validation

```python
from effimemo.adapters import OpenAIAdapter

# Validate message format
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

if OpenAIAdapter.validate_messages(messages):
    compressed = manager.compress(messages)
else:
    print("Invalid message format")
```

### Tool Calling Support

```python
# Support messages with tool calls
messages_with_tools = [
    {"role": "user", "content": "What's the weather like today?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing"}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "Beijing is sunny today, temperature 25°C"
    },
    {"role": "assistant", "content": "Today Beijing has great weather, sunny with a temperature of 25°C."}
]

# Normal compression, tool calls are handled correctly
compressed = manager.compress(messages_with_tools)
```

## API Reference

### create_context_manager

Convenience function to create a context manager instance.

**Parameters:**
- `max_tokens` (int): Maximum token count, default 8192
- `model_name` (str): Model name, default "gpt-4"
- `strategy` (str): Compression strategy, options: "first", "last", "selective", "summary", default "last"
- `preserve_system` (bool): Whether to preserve system messages, default True

**Returns:**
- `ContextManager`: Context manager instance

### ContextManager

Main context management class.

#### Initialization Parameters

- `max_tokens` (int): Maximum token count
- `model_name` (str): Model name
- `strategy` (str): Compression strategy
- `preserve_system` (bool): Whether to preserve system messages
- `token_counter`: Custom token counter
- `openai_client`: OpenAI client instance (for summary strategy)
- `summary_model` (str): Summary model name, default "gpt-3.5-turbo"
- `preserve_recent` (int): Number of recent messages to preserve, default 3
- `summary_prompt` (str): Custom summary prompt
- `min_content_tokens` (int): Minimum content token count, default 100

#### Main Methods

##### compress(messages, reserve_tokens=0)
Compress message list to fit context window.

**Parameters:**
- `messages` (list): Message list
- `reserve_tokens` (int): Tokens to reserve for response

**Returns:**
- `list`: Compressed message list

##### count_tokens(messages)
Count tokens in message list.

**Parameters:**
- `messages` (list): Message list

**Returns:**
- `int`: Token count

## Performance Comparison

Characteristics of different strategies:

| Strategy | Advantages | Disadvantages | Use Cases |
|----------|------------|---------------|-----------|
| **last** | Maintains conversation continuity | May lose important historical information | General conversation scenarios |
| **first** | Preserves initial context | May lose latest information | Scenarios requiring initial settings |
| **selective** | Intelligent content compression | Requires additional dependencies | Content-intensive conversations |
| **summary** | Preserves key information | Requires API calls, has latency | Long-term conversation memory |

## Development Guide

### Environment Setup

```bash
# Clone repository
git clone https://github.com/BetterAndBetterII/effimemo.git
cd effimemo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e .[dev,openai,compression]
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=effimemo --cov-report=html

# Run specific test file
pytest tests/test_manager.py

# Run specific test
pytest tests/test_manager.py::TestContextManager::test_first_strategy
```

### Code Quality Checks

```bash
# Code formatting
black .
isort .

# Code style checks
flake8 effimemo tests

# Run all quality checks
black . && isort . && flake8 effimemo tests && pytest --cov=effimemo
```

### Test Coverage

Current test coverage: **80%+**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

betterandbetterii
