"""
Token计数器模块，负责精确计算不同模型的token数量
"""


class TokenCounter:
    """Token计数器抽象基类"""

    def count(self, text: str) -> int:
        """
        计算文本的token数量

        Args:
            text: 要计算的文本

        Returns:
            token数量
        """
        raise NotImplementedError

    def count_messages(self, messages: list) -> int:
        """
        计算消息列表的token数量

        Args:
            messages: 消息列表

        Returns:
            token数量
        """
        raise NotImplementedError


class TiktokenCounter(TokenCounter):
    """基于tiktoken的token计数器"""

    def __init__(self, model_name: str = "gpt-4"):
        """
        初始化token计数器

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.encoding = self._get_encoding()

    def _get_encoding(self):
        """获取对应模型的编码器"""
        import tiktoken

        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """
        计算文本的token数量

        Args:
            text: 要计算的文本

        Returns:
            token数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: list) -> int:
        """
        计算消息列表的token数量，支持OpenAI格式

        Args:
            messages: 消息列表

        Returns:
            token数量
        """
        if not messages:
            return 0

        total = 0
        for message in messages:
            total += 4  # 每条消息的基础token数

            # 处理不同类型的消息对象
            if hasattr(message, "items"):
                # 类似dict的对象（包括dict和我们的模拟对象）
                items = message.items()
            elif hasattr(message, "model_dump"):
                # Pydantic对象
                items = message.model_dump().items()
            elif hasattr(message, "__dict__"):
                # 普通对象，使用__dict__
                items = message.__dict__.items()
            else:
                # 假设是dict
                items = message.items()

            for key, value in items:
                if key == "tool_calls" and isinstance(value, list):
                    # 处理工具调用
                    for tool_call in value:
                        if isinstance(tool_call, dict):
                            for tc_key, tc_value in tool_call.items():
                                if tc_key == "function" and isinstance(tc_value, dict):
                                    # 处理function字段
                                    for f_key, f_value in tc_value.items():
                                        total += self.count(str(f_value))
                                else:
                                    total += self.count(str(tc_value))
                elif key == "tool_call_id":
                    # 工具调用ID
                    total += self.count(str(value))
                else:
                    # 普通字段
                    total += self.count(str(value))
                if key == "name":
                    total -= 1  # name字段有特殊处理
        total += 2  # 消息格式的额外token
        return total


class CachedTokenCounter(TokenCounter):
    """带缓存的Token计数器"""

    def __init__(self, base_counter: TokenCounter):
        """
        初始化带缓存的token计数器

        Args:
            base_counter: 基础token计数器
        """
        self.base_counter = base_counter
        self.cache = {}

    def count(self, text: str) -> int:
        """
        计算文本的token数量，使用缓存提高性能

        Args:
            text: 要计算的文本

        Returns:
            token数量
        """
        if not text:
            return 0

        if text in self.cache:
            return self.cache[text]
        count = self.base_counter.count(text)
        self.cache[text] = count
        return count

    def count_messages(self, messages: list) -> int:
        """
        计算消息列表的token数量

        Args:
            messages: 消息列表

        Returns:
            token数量
        """
        # 消息列表不可哈希，无法直接缓存
        # 但可以缓存单个消息的内容
        if not messages:
            return 0

        total = 0
        for message in messages:
            message_str = str(message)
            if message_str in self.cache:
                total += self.cache[message_str]
                continue

            # 计算单个消息的token数
            message_tokens = 4  # 每条消息的基础token数

            # 处理不同类型的消息对象
            if hasattr(message, "items"):
                # 类似dict的对象（包括dict和我们的模拟对象）
                items = message.items()
            elif hasattr(message, "model_dump"):
                # Pydantic对象
                items = message.model_dump().items()
            elif hasattr(message, "__dict__"):
                # 普通对象，使用__dict__
                items = message.__dict__.items()
            else:
                # 假设是dict
                items = message.items()

            for key, value in items:
                if key == "tool_calls" and isinstance(value, list):
                    # 处理工具调用
                    for tool_call in value:
                        if isinstance(tool_call, dict):
                            for tc_key, tc_value in tool_call.items():
                                if tc_key == "function" and isinstance(tc_value, dict):
                                    # 处理function字段
                                    for f_key, f_value in tc_value.items():
                                        message_tokens += self.count(str(f_value))
                                else:
                                    message_tokens += self.count(str(tc_value))
                elif key == "tool_call_id":
                    # 工具调用ID
                    message_tokens += self.count(str(value))
                else:
                    # 普通字段
                    message_tokens += self.count(str(value))
                if key == "name":
                    message_tokens -= 1  # name字段有特殊处理

            self.cache[message_str] = message_tokens
            total += message_tokens

        total += 2  # 消息格式的额外token
        return total
