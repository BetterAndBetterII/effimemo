# EffiMemo

[![PyPI version](https://badge.fury.io/py/effimemo.svg)](https://badge.fury.io/py/effimemo)
[![Python Support](https://img.shields.io/pypi/pyversions/effimemo.svg)](https://pypi.org/project/effimemo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/BetterAndBetterII/effimemo/workflows/Tests/badge.svg)](https://github.com/BetterAndBetterII/effimemo/actions)
[![Coverage](https://img.shields.io/badge/coverage-80%25-green.svg)](https://github.com/BetterAndBetterII/effimemo)

一个用于管理大语言模型（LLM）上下文窗口的Python包，支持智能压缩和多种裁切策略。

## 功能特性

- **智能上下文管理**：自动管理对话历史，确保不超过token限制
- **多种压缩策略**：支持 `first`、`last`、`selective` 和 `summary` 四种策略
- **灵活配置**：可自定义最大token数、模型类型等参数
- **系统消息保护**：可选择性保留重要的系统消息
- **OpenAI集成**：支持OpenAI API进行智能摘要压缩
- **工具调用支持**：完整支持OpenAI的function calling和tool使用
- **高测试覆盖率**：80%+ 的测试覆盖率，确保代码质量

## 安装

### 基础安装
```bash
pip install effimemo
```

### 可选依赖
```bash
# 支持OpenAI摘要策略
pip install effimemo[openai]

# 支持selective压缩策略
pip install effimemo[compression]

# 开发依赖
pip install effimemo[dev]

# 安装所有依赖
pip install effimemo[openai,compression,dev]
```

## 快速开始

### 基础用法

```python
from effimemo import create_context_manager

# 创建上下文管理器
manager = create_context_manager(
    max_tokens=8192,
    model_name="gpt-4",
    strategy="last",
    preserve_system=True
)

# 使用管理器处理对话
messages = [
    {"role": "system", "content": "你是一个有用的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "请告诉我关于量子物理的知识"}
]

# 压缩上下文（为响应预留1000个token）
compressed_messages = manager.compress(messages, reserve_tokens=1000)

# 计算token数量
token_count = manager.count_tokens(messages)
print(f"原始消息token数: {token_count}")
```

### 直接使用ContextManager

```python
from effimemo import ContextManager

# 创建管理器实例
manager = ContextManager(
    max_tokens=4096,
    model_name="gpt-3.5-turbo",
    strategy="summary",
    preserve_system=True
)

# 处理长对话
long_conversation = [
    {"role": "system", "content": "你是一个专业的编程助手"},
    # ... 很多对话消息
]

# 压缩对话
result = manager.compress(long_conversation)
```

## 压缩策略详解

### 1. Last策略 (默认)
保留最近的消息，删除较早的消息：
```python
manager = create_context_manager(strategy="last")
```

### 2. First策略
保留最早的消息，删除较新的消息：
```python
manager = create_context_manager(strategy="first")
```

### 3. Selective策略
使用智能压缩算法减少消息内容：
```python
# 需要安装: pip install effimemo[compression]
manager = create_context_manager(strategy="selective")
```

### 4. Summary策略
使用OpenAI API生成对话摘要：
```python
import openai

# 需要安装: pip install effimemo[openai]
client = openai.OpenAI(api_key="your-api-key")

manager = create_context_manager(
    strategy="summary",
    openai_client=client,
    summary_model="gpt-3.5-turbo",
    preserve_recent=3,  # 保留最近3条消息
    summary_prompt="请简洁地总结以下对话内容：\n{conversation}"
)
```

## 高级用法

### 自定义参数

```python
from effimemo import ContextManager

manager = ContextManager(
    max_tokens=8192,
    model_name="gpt-4",
    strategy="summary",
    preserve_system=True,
    # Summary策略参数
    openai_client=openai_client,
    summary_model="gpt-4",
    preserve_recent=5,
    summary_prompt="自定义摘要提示词：{conversation}",
    # 截断策略参数
    min_content_tokens=50
)
```

### 消息验证

```python
from effimemo.adapters import OpenAIAdapter

# 验证消息格式
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

if OpenAIAdapter.validate_messages(messages):
    compressed = manager.compress(messages)
else:
    print("消息格式不正确")
```

### 工具调用支持

```python
# 支持包含工具调用的消息
messages_with_tools = [
    {"role": "user", "content": "今天天气怎么样？"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "北京"}'
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "北京今天晴天，气温25°C"
    },
    {"role": "assistant", "content": "今天北京天气很好，晴天，气温25°C。"}
]

# 正常压缩，工具调用会被正确处理
compressed = manager.compress(messages_with_tools)
```

## API参考

### create_context_manager

创建上下文管理器实例的便捷函数。

**参数：**
- `max_tokens` (int): 最大token数量，默认8192
- `model_name` (str): 模型名称，默认"gpt-4"
- `strategy` (str): 压缩策略，可选"first"、"last"、"selective"或"summary"，默认"last"
- `preserve_system` (bool): 是否保留系统消息，默认True

**返回：**
- `ContextManager`: 上下文管理器实例

### ContextManager

主要的上下文管理类。

#### 初始化参数

- `max_tokens` (int): 最大token数量
- `model_name` (str): 模型名称
- `strategy` (str): 压缩策略
- `preserve_system` (bool): 是否保留系统消息
- `token_counter`: 自定义token计数器
- `openai_client`: OpenAI客户端实例（用于summary策略）
- `summary_model` (str): 摘要模型名称，默认"gpt-3.5-turbo"
- `preserve_recent` (int): 保留最近消息数量，默认3
- `summary_prompt` (str): 自定义摘要提示词
- `min_content_tokens` (int): 最小内容token数量，默认100

#### 主要方法

##### compress(messages, reserve_tokens=0)
压缩消息列表以适应上下文窗口。

**参数：**
- `messages` (list): 消息列表
- `reserve_tokens` (int): 为响应预留的token数量

**返回：**
- `list`: 压缩后的消息列表

##### count_tokens(messages)
计算消息列表的token数量。

**参数：**
- `messages` (list): 消息列表

**返回：**
- `int`: token数量

## 性能对比

不同策略的特点：

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **last** | 保持对话连续性 | 可能丢失重要历史信息 | 一般对话场景 |
| **first** | 保留初始上下文 | 可能丢失最新信息 | 需要保持初始设定的场景 |
| **selective** | 智能内容压缩 | 需要额外依赖 | 内容密集型对话 |
| **summary** | 保留关键信息 | 需要API调用，有延迟 | 长期对话记忆 |

## 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/BetterAndBetterII/effimemo.git
cd effimemo

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e .[dev,openai,compression]
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=effimemo --cov-report=html

# 运行特定测试文件
pytest tests/test_manager.py

# 运行特定测试
pytest tests/test_manager.py::TestContextManager::test_first_strategy
```

### 代码质量检查

```bash
# 代码格式化
black .
isort .

# 代码风格检查
flake8 effimemo tests

# 运行所有质量检查
black . && isort . && flake8 effimemo tests && pytest --cov=effimemo
```

### 测试覆盖率

当前测试覆盖率：**80%+**

主要测试模块：
- ✅ **Context Manager** - 核心管理器功能
- ✅ **Truncation Strategies** - 截断策略（first/last）
- ✅ **Summary Strategy** - 摘要压缩策略
- ✅ **Compression Strategy** - 选择性压缩策略
- ✅ **Token Counter** - Token计数功能
- ✅ **OpenAI Adapter** - OpenAI集成适配器

### 性能测试

项目包含完整的性能比较测试，可以评估不同策略的压缩效果：

```bash
# 运行压缩策略比较测试
pytest tests/test_compression_comparison.py::TestCompressionComparison::test_compression_comparison -v
```

测试结果示例：
- **Summary策略**：压缩率最高（~98%），适合长对话历史
- **Last策略**：保留最新信息（~61%），适合连续对话
- **First策略**：保留早期信息（~64%），适合保持上下文完整性

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

### 提交前检查清单

- [ ] 代码通过所有测试
- [ ] 新功能包含相应测试
- [ ] 代码符合项目风格规范
- [ ] 更新了相关文档
- [ ] 测试覆盖率不低于80%

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 更新日志

### v0.1.1 (2024-01-XX)
- 🐛 修复版本号同步问题
- 📝 完善README文档和API示例
- ✅ 提高测试覆盖率至80%+
- 🔧 优化项目配置和构建流程

### v0.1.0 (2024-01-XX)
- 🎉 首次发布
- ✨ 支持四种压缩策略（first/last/selective/summary）
- 🔧 OpenAI API集成
- 📦 完整的工具调用支持

## 作者

betterandbetterii - betterandbetterii@gmail.com 