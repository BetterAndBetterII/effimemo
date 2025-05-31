"""
上下文管理器主类
"""

from .core.tokenizer import TiktokenCounter
from .strategies.truncation import FirstTruncationStrategy, LastTruncationStrategy
from .strategies.compression import SelectiveCompressionStrategy
from .strategies.summary import SummaryCompressionStrategy


class ContextManager:
    """LLM上下文管理器"""

    def __init__(
        self,
        max_tokens: int = 8192,
        model_name: str = "gpt-4",
        strategy: str = "last",
        preserve_system: bool = True,
        token_counter=None,
        # Summary策略相关参数
        openai_client=None,
        summary_model: str = "gpt-3.5-turbo",
        preserve_recent: int = 3,
        summary_prompt: str = None,
        # 截断策略相关参数
        min_content_tokens: int = 100,
    ):
        """
        初始化上下文管理器

        Args:
            max_tokens: 最大token数量
            model_name: 模型名称
            strategy: 裁切策略，'first'、'last'、'selective'或'summary'
            preserve_system: 是否保留系统消息
            token_counter: 自定义token计数器
            openai_client: OpenAI客户端实例（用于summary策略）
            summary_model: 用于摘要的模型名称
            preserve_recent: 保留最近的消息数量（用于summary策略）
            summary_prompt: 自定义摘要提示词（用于summary策略）
            min_content_tokens: 保留内容的最小token数量（用于截断策略）
        """
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.preserve_system = preserve_system
        self.token_counter = token_counter or TiktokenCounter(model_name)

        # 根据策略名称选择策略
        if strategy == "first":
            self.strategy = FirstTruncationStrategy(
                preserve_system=preserve_system, min_content_tokens=min_content_tokens
            )
        elif strategy == "last":
            self.strategy = LastTruncationStrategy(
                preserve_system=preserve_system, min_content_tokens=min_content_tokens
            )
        elif strategy == "selective":
            self.strategy = SelectiveCompressionStrategy(
                preserve_system=preserve_system
            )
        elif strategy == "summary":
            self.strategy = SummaryCompressionStrategy(
                openai_client=openai_client,
                model=summary_model,
                preserve_system=preserve_system,
                preserve_recent=preserve_recent,
                summary_prompt=summary_prompt,
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
