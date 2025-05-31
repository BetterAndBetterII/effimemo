"""
上下文管理器主类
"""

from .core.tokenizer import TiktokenCounter
from .strategies.truncation import FirstTruncationStrategy, LastTruncationStrategy
from .strategies.compression import SelectiveCompressionStrategy


class ContextManager:
    """LLM上下文管理器"""

    def __init__(
        self,
        max_tokens: int = 8192,
        model_name: str = "gpt-4",
        strategy: str = "last",
        preserve_system: bool = True,
        token_counter=None,
    ):
        """
        初始化上下文管理器

        Args:
            max_tokens: 最大token数量
            model_name: 模型名称
            strategy: 裁切策略，'first'、'last'或'selective'
            preserve_system: 是否保留系统消息
            token_counter: 自定义token计数器
        """
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.preserve_system = preserve_system
        self.token_counter = token_counter or TiktokenCounter(model_name)

        # 根据策略名称选择策略
        if strategy == "first":
            self.strategy = FirstTruncationStrategy(preserve_system=preserve_system)
        elif strategy == "last":
            self.strategy = LastTruncationStrategy(preserve_system=preserve_system)
        elif strategy == "selective":
            self.strategy = SelectiveCompressionStrategy(
                preserve_system=preserve_system
            )
        else:
            raise ValueError(f"不支持的策略: {strategy}")

    def compress(self, messages: list, reserve_tokens: int = 0) -> list:
        """
        压缩消息列表以适应上下文窗口

        Args:
            messages: 消息列表
            reserve_tokens: 为响应预留的token数量

        Returns:
            处理后的消息列表
        """
        available_tokens = self.max_tokens - reserve_tokens
        return self.strategy.compress(messages, available_tokens, self.token_counter)

    def count_tokens(self, messages: list) -> int:
        """
        计算消息列表的token数量

        Args:
            messages: 消息列表

        Returns:
            token数量
        """
        return self.token_counter.count_messages(messages)
