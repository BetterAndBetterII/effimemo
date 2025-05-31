"""
OpenAI适配器模块
"""


class OpenAIAdapter:
    """OpenAI接口适配器"""

    @staticmethod
    def is_valid_message(message: dict) -> bool:
        """
        检查消息格式是否有效

        Args:
            message: 消息字典

        Returns:
            是否有效
        """
        if not isinstance(message, dict):
            return False

        if "role" not in message:
            return False

        role = message["role"]
        if role not in ["system", "user", "assistant", "tool", "function"]:
            return False

        if role in ["user", "system", "assistant"] and "content" not in message:
            # 助手消息可能没有content，但有tool_calls
            if role == "assistant" and "tool_calls" in message:
                return True
            return False

        if role == "tool" and (
            "content" not in message or "tool_call_id" not in message
        ):
            return False

        return True

    @staticmethod
    def validate_messages(messages: list) -> bool:
        """
        验证消息列表格式是否正确

        Args:
            messages: 消息列表

        Returns:
            格式是否正确
        """
        if not isinstance(messages, list):
            return False

        for message in messages:
            if not OpenAIAdapter.is_valid_message(message):
                return False

        return True

    @staticmethod
    def extract_tool_calls(messages: list) -> list:
        """
        提取所有工具调用

        Args:
            messages: 消息列表

        Returns:
            工具调用列表
        """
        tool_calls = []
        for message in messages:
            if message.get("role") == "assistant" and "tool_calls" in message:
                tool_calls.extend(message["tool_calls"])
        return tool_calls

    @staticmethod
    def extract_tool_results(messages: list) -> list:
        """
        提取所有工具调用结果

        Args:
            messages: 消息列表

        Returns:
            工具调用结果列表
        """
        tool_results = []
        for message in messages:
            if message.get("role") == "tool":
                tool_results.append(message)
        return tool_results
