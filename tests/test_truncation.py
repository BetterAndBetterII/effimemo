import pytest
from effimemo.strategies.truncation import (
    FirstTruncationStrategy,
    LastTruncationStrategy,
)
from effimemo.core.tokenizer import TiktokenCounter


class TestTruncationStrategies:
    def setup_method(self):
        self.token_counter = TiktokenCounter()
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "Tell me about quantum physics."},
            {
                "role": "assistant",
                "content": "Quantum physics is a branch of physics that deals with the behavior of matter and energy at the smallest scales.",
            },
        ]

    def test_first_truncation_empty_messages(self):
        strategy = FirstTruncationStrategy()
        result = strategy.compress([], 1000, self.token_counter)
        assert result == []

    def test_first_truncation_under_limit(self):
        strategy = FirstTruncationStrategy()
        # Set a high limit that won't trigger truncation
        result = strategy.compress(self.messages, 1000, self.token_counter)
        assert len(result) == len(self.messages)
        assert result == self.messages

    def test_first_truncation_over_limit(self):
        strategy = FirstTruncationStrategy()
        # Set a low limit that will trigger truncation
        # Should keep system message and first few messages
        result = strategy.compress(self.messages, 30, self.token_counter)
        assert len(result) < len(self.messages)
        assert result[0] == self.messages[0]  # System message preserved

    def test_first_truncation_no_preserve_system(self):
        strategy = FirstTruncationStrategy(preserve_system=False)
        result = strategy.compress(self.messages, 30, self.token_counter)
        assert len(result) < len(self.messages)
        # Should just keep first few messages regardless of role

    def test_first_truncation_content_truncation_string(self):
        """测试FirstTruncationStrategy的字符串内容截断"""
        strategy = FirstTruncationStrategy()
        long_content = "This is a very long message that should be truncated. " * 20
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": long_content},
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 应该有截断的用户消息
        user_messages = [m for m in result if m.get("role") == "user"]
        if user_messages:
            assert len(user_messages[0]["content"]) < len(long_content)

    def test_first_truncation_system_message_overflow(self):
        """测试FirstTruncationStrategy的系统消息超出限制处理"""
        strategy = FirstTruncationStrategy()
        long_system_content = "You are a very helpful assistant. " * 100
        messages = [
            {"role": "system", "content": long_system_content},
            {"role": "user", "content": "Hello!"},
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 应该只有一条截断的系统消息
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) < len(long_system_content)

    def test_first_truncation_tool_call_handling(self):
        """测试FirstTruncationStrategy的工具调用消息处理"""
        strategy = FirstTruncationStrategy()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "function": {"name": "test"}}],
            },
            {"role": "user", "content": "Hello!"},
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 工具调用消息应该被保留
        tool_messages = [m for m in result if m.get("tool_calls")]
        assert len(tool_messages) > 0

    def test_first_truncation_developer_role(self):
        """测试FirstTruncationStrategy对developer角色的处理"""
        strategy = FirstTruncationStrategy()
        messages = [
            {"role": "developer", "content": "Developer instructions."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = strategy.compress(messages, 30, self.token_counter)

        # developer消息应该被保留
        dev_messages = [m for m in result if m.get("role") == "developer"]
        assert len(dev_messages) > 0

    def test_last_truncation_empty_messages(self):
        strategy = LastTruncationStrategy()
        result = strategy.compress([], 1000, self.token_counter)
        assert result == []

    def test_last_truncation_under_limit(self):
        strategy = LastTruncationStrategy()
        # Set a high limit that won't trigger truncation
        result = strategy.compress(self.messages, 1000, self.token_counter)
        assert len(result) == len(self.messages)
        assert result == self.messages

    def test_last_truncation_over_limit(self):
        strategy = LastTruncationStrategy()
        # Set a low limit that will trigger truncation
        # Should keep system message and last few messages
        result = strategy.compress(self.messages, 30, self.token_counter)
        assert len(result) < len(self.messages)
        assert result[0] == self.messages[0]  # System message preserved

    def test_last_truncation_no_preserve_system(self):
        strategy = LastTruncationStrategy(preserve_system=False)
        result = strategy.compress(self.messages, 30, self.token_counter)
        assert len(result) < len(self.messages)
        # Should just keep last few messages regardless of role

    def test_last_truncation_content_truncation_string(self):
        """测试LastTruncationStrategy的字符串内容截断"""
        strategy = LastTruncationStrategy()
        long_content = "This is a very long message that should be truncated. " * 20
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": long_content},
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 应该有截断的用户消息
        user_messages = [m for m in result if m.get("role") == "user"]
        if user_messages:
            assert len(user_messages[0]["content"]) < len(long_content)

    def test_last_truncation_content_truncation_list(self):
        """测试LastTruncationStrategy的列表内容截断"""
        strategy = LastTruncationStrategy()
        long_text = "This is a very long text content. " * 20
        messages = [
            {"role": "system", "content": "System message."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": long_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/image.jpg"},
                    },
                ],
            },
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 应该有截断的用户消息
        user_messages = [m for m in result if m.get("role") == "user"]
        if user_messages:
            content = user_messages[0]["content"]
            assert isinstance(content, list)
            # 应该保留图片部分
            image_parts = [c for c in content if c.get("type") == "image_url"]
            assert len(image_parts) > 0

    def test_last_truncation_system_message_overflow(self):
        """测试LastTruncationStrategy的系统消息超出限制处理"""
        strategy = LastTruncationStrategy()
        long_system_content = "You are a very helpful assistant. " * 100
        messages = [
            {"role": "system", "content": long_system_content},
            {"role": "user", "content": "Hello!"},
        ]
        result = strategy.compress(messages, 50, self.token_counter)

        # 应该只有一条截断的系统消息
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert len(result[0]["content"]) < len(long_system_content)

    def test_last_truncation_min_content_tokens_threshold(self):
        """测试LastTruncationStrategy的最小内容token阈值"""
        strategy = LastTruncationStrategy(min_content_tokens=200)
        long_content = "This is a long message. " * 50
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": long_content},
        ]
        result = strategy.compress(messages, 150, self.token_counter)

        # 由于剩余空间不足min_content_tokens，应该不会尝试截断
        user_messages = [m for m in result if m.get("role") == "user"]
        # 可能没有用户消息，或者有完整的用户消息
        if user_messages:
            assert user_messages[0]["content"] == long_content

    def test_message_order_preservation(self):
        """测试消息顺序保持"""
        strategy = LastTruncationStrategy()
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "First user message."},
            {"role": "assistant", "content": "First assistant message."},
            {"role": "user", "content": "Second user message."},
            {"role": "assistant", "content": "Second assistant message."},
        ]
        result = strategy.compress(messages, 100, self.token_counter)

        # 检查消息顺序
        prev_index = -1
        for message in result:
            if message in messages:
                current_index = messages.index(message)
                assert current_index > prev_index, "消息顺序应该保持"
                prev_index = current_index
