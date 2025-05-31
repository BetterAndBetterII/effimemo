"""
包入口点和高级API
"""

from .manager import ContextManager

# 导出主要类
__all__ = ["ContextManager", "create_context_manager"]


def create_context_manager(
    max_tokens: int = 8192,
    model_name: str = "gpt-4",
    strategy: str = "last",
    preserve_system: bool = True,
) -> ContextManager:
    """
    创建上下文管理器

    Args:
        max_tokens: 最大token数量
        model_name: 模型名称
        strategy: 裁切策略，'first'、'last'或'selective'
        preserve_system: 是否保留系统消息

    Returns:
        上下文管理器实例
    """
    return ContextManager(
        max_tokens=max_tokens,
        model_name=model_name,
        strategy=strategy,
        preserve_system=preserve_system,
    )
