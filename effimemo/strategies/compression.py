"""
压缩策略实现模块
"""

from .base import ContextStrategy
from .truncation import LastTruncationStrategy


class SelectiveCompressionStrategy(ContextStrategy):
    """基于Selective_Context的压缩策略"""

    def __init__(
        self, reduce_ratio: float = 0.5, lang: str = "en", preserve_system: bool = True
    ):
        """
        初始化压缩策略

        Args:
            reduce_ratio: 压缩比例
            lang: 语言
            preserve_system: 是否保留系统消息
        """
        self.reduce_ratio = reduce_ratio
        self.lang = lang
        self.preserve_system = preserve_system
        self._init_selective_context()

    def _init_selective_context(self):
        """初始化Selective_Context"""
        try:
            from selective_context import SelectiveContext

            self.sc = SelectiveContext(model_type="gpt2", lang=self.lang)
        except ImportError:
            # 如果没有安装selective_context，使用简单的压缩方法
            self.sc = None
            print("警告: selective-context未安装，将使用简单的压缩方法")

    def _simple_compress(self, text: str) -> str:
        """
        简单的压缩方法，当selective_context未安装时使用

        Args:
            text: 要压缩的文本

        Returns:
            压缩后的文本
        """
        if not text:
            return ""

        # 简单的压缩方法：保留前面的部分
        words = text.split()
        keep_count = max(1, int(len(words) * (1 - self.reduce_ratio)))
        return " ".join(words[:keep_count])

    def compress(self, messages: list, max_tokens: int, token_counter) -> list:
        """
        压缩消息列表

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            压缩后的消息列表
        """
        if not messages:
            return []

        # 先尝试压缩内容
        result = []
        for message in messages:
            if self.preserve_system and message.get("role") == "system":
                # 系统消息不压缩
                result.append(message)
                continue

            new_message = message.copy()

            # 处理普通内容
            content = message.get("content", "")
            if content:
                if self.sc:
                    compressed, _ = self.sc(content, reduce_ratio=self.reduce_ratio)
                    new_message["content"] = compressed
                else:
                    new_message["content"] = self._simple_compress(content)

            # 处理工具调用结果
            if "tool_call_id" in message and "content" in message:
                # 这是工具调用的结果，也需要压缩
                if self.sc:
                    compressed, _ = self.sc(
                        message["content"], reduce_ratio=self.reduce_ratio
                    )
                    new_message["content"] = compressed
                else:
                    new_message["content"] = self._simple_compress(message["content"])

            result.append(new_message)

        # 计算压缩后的token数
        compressed_tokens = token_counter.count_messages(result)

        # 如果压缩后仍然超出限制，使用LastTruncationStrategy进一步裁切
        if compressed_tokens > max_tokens:
            truncator = LastTruncationStrategy(preserve_system=self.preserve_system)
            return truncator.compress(result, max_tokens, token_counter)

        return result
