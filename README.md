# EffiMemo

一个用于管理大语言模型（LLM）上下文窗口的Python包。

## 功能特性

- 智能上下文管理：自动管理对话历史，确保不超过token限制
- 多种裁切策略：支持first、last和selective策略
- 灵活配置：可自定义最大token数、模型类型等参数
- 系统消息保护：可选择性保留重要的系统消息

## 安装

```bash
pip install effimemo
```

## 快速开始

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
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
]

# 管理上下文
managed_messages = manager.manage_context(messages)
```

## API文档

### create_context_manager

创建上下文管理器实例。

**参数：**
- `max_tokens` (int): 最大token数量，默认8192
- `model_name` (str): 模型名称，默认"gpt-4"
- `strategy` (str): 裁切策略，可选"first"、"last"或"selective"，默认"last"
- `preserve_system` (bool): 是否保留系统消息，默认True

**返回：**
- `ContextManager`: 上下文管理器实例

## 许可证

MIT License

## 作者

Manus AI 