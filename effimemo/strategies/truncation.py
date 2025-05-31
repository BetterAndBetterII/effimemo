"""
裁切策略实现模块
"""

from .base import ContextStrategy


class FirstTruncationStrategy(ContextStrategy):
    """保留前面消息的裁切策略，支持内容截断"""

    def __init__(self, preserve_system: bool = True, min_content_tokens: int = 100):
        """
        初始化裁切策略

        Args:
            preserve_system: 是否保留系统消息
            min_content_tokens: 保留内容的最小token数量
        """
        self.preserve_system = preserve_system
        self.min_content_tokens = min_content_tokens

    def _truncate_string(self, text: str, max_tokens: int, token_counter) -> str:
        """
        截断字符串到指定的token数量

        Args:
            text: 要截断的文本
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            截断后的文本
        """
        if not text:
            return text

        # 如果文本已经在限制范围内，直接返回
        current_tokens = token_counter.count(text)
        if current_tokens <= max_tokens:
            return text

        # 二分查找最佳截断位置
        left, right = 0, len(text)
        best_text = ""

        while left <= right:
            mid = (left + right) // 2
            truncated = text[:mid]
            tokens = token_counter.count(truncated)

            if tokens <= max_tokens:
                best_text = truncated
                left = mid + 1
            else:
                right = mid - 1

        return best_text

    def _handle_system_message_overflow(
        self, system_messages: list, max_tokens: int, token_counter
    ) -> list:
        """
        处理系统消息超出限制的情况

        Args:
            system_messages: 系统消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            处理后的系统消息列表
        """
        if not system_messages:
            return []

        # 只保留第一条系统消息
        first_message = system_messages[0].copy()
        content = first_message.get("content", "")

        if isinstance(content, str):
            # 字符串内容直接截断
            truncated_content = self._truncate_string(
                content, max_tokens, token_counter
            )
            first_message["content"] = truncated_content
        elif isinstance(content, list):
            # 列表内容，合并文本部分后截断
            text_parts = []
            other_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    other_parts.append(item)

            # 合并文本并截断
            combined_text = "\n".join(text_parts)
            truncated_text = self._truncate_string(
                combined_text, max_tokens, token_counter
            )

            # 重新构建content
            new_content = other_parts.copy()
            if truncated_text:
                new_content.append({"type": "text", "text": truncated_text})

            first_message["content"] = new_content

        return [first_message]

    def _try_truncate_message(
        self, message: dict, remaining_tokens: int, token_counter
    ) -> tuple:
        """
        尝试截断单个消息

        Args:
            message: 要截断的消息
            remaining_tokens: 剩余token数量
            token_counter: token计数器

        Returns:
            (截断后的消息, 是否成功截断)
        """
        # 如果剩余空间不足，不进行截断
        if remaining_tokens <= self.min_content_tokens:
            return None, False

        # 跳过工具调用消息和没有content的消息
        if message.get("role") == "tool" or "content" not in message:
            return None, False

        # 跳过assistant的工具调用消息（只有tool_calls字段计费）
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return message.copy(), True

        truncated_message = message.copy()
        content = message["content"]

        if isinstance(content, str):
            # 字符串内容截断
            truncated_content = self._truncate_string(
                content, remaining_tokens, token_counter
            )
            truncated_message["content"] = truncated_content

            # 检查截断后的消息是否符合要求
            message_tokens = token_counter.count_messages([truncated_message])
            if message_tokens <= remaining_tokens:
                return truncated_message, True

        elif isinstance(content, list):
            # 列表内容处理
            text_parts = []
            other_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    other_parts.append(item)

            # 合并文本部分
            combined_text = "\n".join(text_parts)
            truncated_text = self._truncate_string(
                combined_text, remaining_tokens, token_counter
            )

            # 重新构建content
            new_content = other_parts.copy()
            if truncated_text:
                new_content.append({"type": "text", "text": truncated_text})

            truncated_message["content"] = new_content

            # 检查截断后的消息是否符合要求
            message_tokens = token_counter.count_messages([truncated_message])
            if message_tokens <= remaining_tokens:
                return truncated_message, True

        return None, False

    def compress(self, messages: list, max_tokens: int, token_counter) -> list:
        """
        保留前面的消息，裁切后面的消息

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            裁切后的消息列表
        """
        if not messages:
            return []

        # 检查当前消息总token数
        current_tokens = token_counter.count_messages(messages)

        # 如果当前token数已经在限制范围内，直接返回
        if current_tokens <= max_tokens:
            return messages

        # 分离系统消息和其他消息
        system_messages = []
        non_system_messages = []

        for message in messages:
            if self.preserve_system and message.get("role") in ["system", "developer"]:
                system_messages.append(message)
            else:
                non_system_messages.append(message)

        # 计算系统消息的token数
        system_tokens = 0
        if system_messages:
            system_tokens = token_counter.count_messages(system_messages)

        # 如果系统消息已经超出限制，进行系统消息截断
        if system_tokens > max_tokens:
            return self._handle_system_message_overflow(
                system_messages, max_tokens, token_counter
            )

        # 如果只有系统消息，直接返回
        if not non_system_messages:
            return system_messages

        # 从前往后处理非系统消息
        result = system_messages.copy()
        preserved_tokens = system_tokens

        for message in non_system_messages:
            message_tokens = token_counter.count_messages([message])

            # 检查添加此消息是否会超出限制
            if preserved_tokens + message_tokens <= max_tokens:
                # 如果不会超出，添加整个消息
                result.append(message)
                preserved_tokens += message_tokens
            else:
                # 如果会超出，尝试截断消息内容
                remaining_tokens = max_tokens - preserved_tokens
                truncated_message, success = self._try_truncate_message(
                    message, remaining_tokens, token_counter
                )

                if success and truncated_message:
                    result.append(truncated_message)
                    preserved_tokens += token_counter.count_messages(
                        [truncated_message]
                    )

                # 无论是否成功截断，都停止处理更多消息
                break

        return result


class LastTruncationStrategy(ContextStrategy):
    """保留后面消息的裁切策略，支持内容截断"""

    def __init__(self, preserve_system: bool = True, min_content_tokens: int = 100):
        """
        初始化裁切策略

        Args:
            preserve_system: 是否保留系统消息
            min_content_tokens: 保留内容的最小token数量
        """
        self.preserve_system = preserve_system
        self.min_content_tokens = min_content_tokens

    def _truncate_string(self, text: str, max_tokens: int, token_counter) -> str:
        """
        截断字符串到指定的token数量

        Args:
            text: 要截断的文本
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            截断后的文本
        """
        if not text:
            return text

        # 如果文本已经在限制范围内，直接返回
        current_tokens = token_counter.count(text)
        if current_tokens <= max_tokens:
            return text

        # 二分查找最佳截断位置
        left, right = 0, len(text)
        best_text = ""

        while left <= right:
            mid = (left + right) // 2
            truncated = text[:mid]
            tokens = token_counter.count(truncated)

            if tokens <= max_tokens:
                best_text = truncated
                left = mid + 1
            else:
                right = mid - 1

        return best_text

    def _handle_system_message_overflow(
        self, system_messages: list, max_tokens: int, token_counter
    ) -> list:
        """
        处理系统消息超出限制的情况

        Args:
            system_messages: 系统消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            处理后的系统消息列表
        """
        if not system_messages:
            return []

        # 只保留第一条系统消息
        first_message = system_messages[0].copy()
        content = first_message.get("content", "")

        if isinstance(content, str):
            # 字符串内容直接截断
            truncated_content = self._truncate_string(
                content, max_tokens, token_counter
            )
            first_message["content"] = truncated_content
        elif isinstance(content, list):
            # 列表内容，合并文本部分后截断
            text_parts = []
            other_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    other_parts.append(item)

            # 合并文本并截断
            combined_text = "\n".join(text_parts)
            truncated_text = self._truncate_string(
                combined_text, max_tokens, token_counter
            )

            # 重新构建content
            new_content = other_parts.copy()
            if truncated_text:
                new_content.append({"type": "text", "text": truncated_text})

            first_message["content"] = new_content

        return [first_message]

    def _try_truncate_message(
        self, message: dict, remaining_tokens: int, token_counter
    ) -> tuple:
        """
        尝试截断单个消息

        Args:
            message: 要截断的消息
            remaining_tokens: 剩余token数量
            token_counter: token计数器

        Returns:
            (截断后的消息, 是否成功截断)
        """
        # 如果剩余空间不足，不进行截断
        if remaining_tokens <= self.min_content_tokens:
            return None, False

        # 跳过工具调用消息和没有content的消息
        if message.get("role") == "tool" or "content" not in message:
            return None, False

        # 跳过assistant的工具调用消息（只有tool_calls字段计费）
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return message.copy(), True

        truncated_message = message.copy()
        content = message["content"]

        if isinstance(content, str):
            # 字符串内容截断
            truncated_content = self._truncate_string(
                content, remaining_tokens, token_counter
            )
            truncated_message["content"] = truncated_content

            # 检查截断后的消息是否符合要求
            message_tokens = token_counter.count_messages([truncated_message])
            if message_tokens <= remaining_tokens:
                return truncated_message, True

        elif isinstance(content, list):
            # 列表内容处理
            text_parts = []
            other_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    other_parts.append(item)

            # 合并文本部分
            combined_text = "\n".join(text_parts)
            truncated_text = self._truncate_string(
                combined_text, remaining_tokens, token_counter
            )

            # 重新构建content
            new_content = other_parts.copy()
            if truncated_text:
                new_content.append({"type": "text", "text": truncated_text})

            truncated_message["content"] = new_content

            # 检查截断后的消息是否符合要求
            message_tokens = token_counter.count_messages([truncated_message])
            if message_tokens <= remaining_tokens:
                return truncated_message, True

        return None, False

    def compress(self, messages: list, max_tokens: int, token_counter) -> list:
        """
        保留后面的消息，裁切前面的消息

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            裁切后的消息列表
        """
        if not messages:
            return []

        # 检查当前消息总token数
        current_tokens = token_counter.count_messages(messages)

        # 如果当前token数已经在限制范围内，直接返回
        if current_tokens <= max_tokens:
            return messages

        # 分离系统消息和其他消息
        system_messages = []
        non_system_messages = []

        for message in messages:
            if self.preserve_system and message.get("role") in ["system", "developer"]:
                system_messages.append(message)
            else:
                non_system_messages.append(message)

        # 计算系统消息的token数
        system_tokens = 0
        if system_messages:
            system_tokens = token_counter.count_messages(system_messages)

        # 如果系统消息已经超出限制，进行系统消息截断
        if system_tokens > max_tokens:
            return self._handle_system_message_overflow(
                system_messages, max_tokens, token_counter
            )

        # 如果只有系统消息，直接返回
        if not non_system_messages:
            return system_messages

        # 从最新消息开始向前处理
        result = system_messages.copy()
        preserved_tokens = system_tokens

        for message in reversed(non_system_messages):
            message_tokens = token_counter.count_messages([message])

            # 检查添加此消息是否会超出限制
            if preserved_tokens + message_tokens <= max_tokens:
                # 如果不会超出，添加整个消息
                result.insert(len(system_messages), message)
                preserved_tokens += message_tokens
            else:
                # 如果会超出，尝试截断消息内容
                remaining_tokens = max_tokens - preserved_tokens
                truncated_message, success = self._try_truncate_message(
                    message, remaining_tokens, token_counter
                )

                if success and truncated_message:
                    result.insert(len(system_messages), truncated_message)
                    preserved_tokens += token_counter.count_messages(
                        [truncated_message]
                    )

                # 无论是否成功截断，都停止处理更多消息
                break

        return result
