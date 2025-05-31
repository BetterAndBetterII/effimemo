"""
摘要压缩策略实现模块
"""

from .base import ContextStrategy
from .truncation import LastTruncationStrategy


class SummaryCompressionStrategy(ContextStrategy):
    """基于OpenAI的摘要压缩策略"""

    def __init__(
        self,
        openai_client=None,
        model: str = "gpt-3.5-turbo",
        preserve_system: bool = True,
        preserve_recent: int = 3,
        summary_prompt: str = None,
    ):
        """
        初始化摘要压缩策略

        Args:
            openai_client: OpenAI客户端实例
            model: 用于摘要的模型名称
            preserve_system: 是否保留系统消息
            preserve_recent: 保留最近的消息数量
            summary_prompt: 自定义摘要提示词
        """
        self.openai_client = openai_client
        self.model = model
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.summary_prompt = summary_prompt or self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """获取默认的摘要提示词"""
        return """请将以下对话历史压缩成简洁的摘要，保留关键信息和上下文：

要求：
1. 保留重要的事实信息和决策
2. 保持对话的逻辑流程
3. 去除冗余和重复内容
4. 使用简洁明了的语言
5. 保持原有的语言风格（中文/英文）

对话历史：
{conversation}

请提供压缩后的摘要："""

    def _format_messages_for_summary(self, messages: list) -> str:
        """
        将消息列表格式化为适合摘要的文本

        Args:
            messages: 消息列表

        Returns:
            格式化后的文本
        """
        formatted_parts = []

        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # 跳过空内容
            if not content:
                continue

            # 处理不同角色的消息
            if role == "system":
                formatted_parts.append(f"系统: {content}")
            elif role == "user":
                formatted_parts.append(f"用户: {content}")
            elif role == "assistant":
                formatted_parts.append(f"助手: {content}")
            elif role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                formatted_parts.append(f"工具结果({tool_call_id}): {content}")
            else:
                formatted_parts.append(f"{role}: {content}")

        return "\n\n".join(formatted_parts)

    def _create_summary(self, messages: list) -> str:
        """
        使用OpenAI客户端创建摘要

        Args:
            messages: 要摘要的消息列表

        Returns:
            摘要文本
        """
        if not self.openai_client:
            # 如果没有OpenAI客户端，使用简单的截断方法
            return self._simple_summary(messages)

        try:
            # 格式化对话历史
            conversation_text = self._format_messages_for_summary(messages)

            # 构建摘要请求
            summary_request = self.summary_prompt.format(conversation=conversation_text)

            # 调用OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_request}],
                temperature=0.3,
                max_tokens=1000,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"警告: OpenAI摘要失败，使用简单摘要方法: {e}")
            return self._simple_summary(messages)

    def _simple_summary(self, messages: list) -> str:
        """
        简单的摘要方法，当OpenAI客户端不可用时使用

        Args:
            messages: 消息列表

        Returns:
            简单摘要
        """
        if not messages:
            return ""

        # 提取关键信息
        key_points = []
        for message in messages:
            content = message.get("content", "")
            if content and len(content) > 10:  # 降低阈值，只跳过很短的内容
                # 截取前100个字符作为摘要
                summary_part = content[:100] + "..." if len(content) > 100 else content
                role = message.get("role", "unknown")
                key_points.append(f"[{role}] {summary_part}")

        return "\n".join(key_points[:5])  # 最多保留5个要点

    def compress(self, messages: list, max_tokens: int, token_counter) -> list:
        """
        使用摘要方式压缩消息列表

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            token_counter: token计数器

        Returns:
            压缩后的消息列表
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
        other_messages = []

        for message in messages:
            if self.preserve_system and message.get("role") == "system":
                system_messages.append(message)
            else:
                other_messages.append(message)

        # 如果只有系统消息，使用截断策略
        if not other_messages:
            truncator = LastTruncationStrategy(preserve_system=self.preserve_system)
            return truncator.compress(messages, max_tokens, token_counter)

        # 保留最近的消息
        recent_messages = (
            other_messages[-self.preserve_recent :]
            if len(other_messages) > self.preserve_recent
            else other_messages
        )

        # 需要摘要的消息
        messages_to_summarize = (
            other_messages[: -self.preserve_recent]
            if len(other_messages) > self.preserve_recent
            else []
        )

        result = system_messages.copy()

        # 如果有需要摘要的消息，创建摘要
        if messages_to_summarize:
            summary_text = self._create_summary(messages_to_summarize)
            if summary_text:
                summary_message = {
                    "role": "system",
                    "content": f"[对话摘要] {summary_text}",
                }
                result.append(summary_message)

        # 添加最近的消息
        result.extend(recent_messages)

        # 检查压缩后的token数
        compressed_tokens = token_counter.count_messages(result)

        # 如果压缩后仍然超出限制，使用LastTruncationStrategy进一步裁切
        if compressed_tokens > max_tokens:
            truncator = LastTruncationStrategy(preserve_system=self.preserve_system)
            return truncator.compress(result, max_tokens, token_counter)

        return result
