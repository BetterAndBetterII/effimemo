"""
裁切策略实现模块
"""

from .base import ContextStrategy


class FirstTruncationStrategy(ContextStrategy):
    """保留前面消息的裁切策略"""

    def __init__(self, preserve_system: bool = True):
        """
        初始化裁切策略

        Args:
            preserve_system: 是否保留系统消息
        """
        self.preserve_system = preserve_system

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

        # 保留系统消息
        if self.preserve_system:
            system_messages = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]

            # 如果只有系统消息，直接返回
            if not non_system:
                return system_messages

            # 计算系统消息的token数
            system_tokens = token_counter.count_messages(system_messages)

            # 如果系统消息已经超出限制，只保留第一条系统消息
            if system_tokens > max_tokens:
                return [system_messages[0]] if system_messages else []

            # 从前往后添加非系统消息
            result = system_messages.copy()
            remaining_tokens = max_tokens - system_tokens

            for message in non_system:
                message_tokens = token_counter.count_messages([message])
                if message_tokens <= remaining_tokens:
                    result.append(message)
                    remaining_tokens -= message_tokens
                else:
                    break

            return result
        else:
            # 不特殊处理系统消息，直接从前往后添加
            result = []
            remaining_tokens = max_tokens

            for message in messages:
                message_tokens = token_counter.count_messages([message])
                if message_tokens <= remaining_tokens:
                    result.append(message)
                    remaining_tokens -= message_tokens
                else:
                    break

            return result


class LastTruncationStrategy(ContextStrategy):
    """保留后面消息的裁切策略"""

    def __init__(self, preserve_system: bool = True):
        """
        初始化裁切策略

        Args:
            preserve_system: 是否保留系统消息
        """
        self.preserve_system = preserve_system

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

        # 保留系统消息
        if self.preserve_system:
            system_messages = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]

            # 如果只有系统消息，直接返回
            if not non_system:
                return system_messages

            # 计算系统消息的token数
            system_tokens = token_counter.count_messages(system_messages)

            # 如果系统消息已经超出限制，只保留第一条系统消息
            if system_tokens > max_tokens:
                return [system_messages[0]] if system_messages else []

            # 从后往前添加非系统消息
            result = system_messages.copy()
            remaining_tokens = max_tokens - system_tokens

            for message in reversed(non_system):
                message_tokens = token_counter.count_messages([message])
                if message_tokens <= remaining_tokens:
                    # 插入到系统消息之后，保持顺序
                    result.insert(len(system_messages), message)
                    remaining_tokens -= message_tokens
                else:
                    break

            return result
        else:
            # 不特殊处理系统消息，直接从后往前添加
            result = []
            remaining_tokens = max_tokens

            for message in reversed(messages):
                message_tokens = token_counter.count_messages([message])
                if message_tokens <= remaining_tokens:
                    result.insert(0, message)
                    remaining_tokens -= message_tokens
                else:
                    break

            return result
