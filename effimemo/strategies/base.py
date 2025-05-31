"""
上下文处理策略抽象基类
"""


class ContextStrategy:
    """上下文处理策略抽象基类"""

    def compress(self, messages: list, max_tokens: int, token_counter) -> list:
        """
        压缩消息列表以适应最大token限制

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            处理后的消息列表
        """
        raise NotImplementedError
