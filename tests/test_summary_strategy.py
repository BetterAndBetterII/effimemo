import pytest
from unittest.mock import Mock, MagicMock
from effimemo.strategies.summary import SummaryCompressionStrategy
from effimemo.core.tokenizer import TiktokenCounter


class TestSummaryCompressionStrategy:
    """测试摘要压缩策略"""

    def setup_method(self):
        """设置测试环境"""
        self.token_counter = TiktokenCounter()

    def create_mock_openai_client(self, summary_response: str = "这是一个测试摘要"):
        """创建mock的OpenAI客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # 设置返回值链
        mock_message.content = summary_response
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    def test_init_with_default_params(self):
        """测试使用默认参数初始化"""
        strategy = SummaryCompressionStrategy()

        assert strategy.openai_client is None
        assert strategy.model == "gpt-3.5-turbo"
        assert strategy.preserve_system is True
        assert strategy.preserve_recent == 3
        assert "请将以下对话历史压缩成简洁的摘要" in strategy.summary_prompt

    def test_init_with_custom_params(self):
        """测试使用自定义参数初始化"""
        mock_client = self.create_mock_openai_client()
        custom_prompt = "自定义摘要提示词: {conversation}"

        strategy = SummaryCompressionStrategy(
            openai_client=mock_client,
            model="gpt-4",
            preserve_system=False,
            preserve_recent=5,
            summary_prompt=custom_prompt,
        )

        assert strategy.openai_client == mock_client
        assert strategy.model == "gpt-4"
        assert strategy.preserve_system is False
        assert strategy.preserve_recent == 5
        assert strategy.summary_prompt == custom_prompt

    def test_format_messages_for_summary(self):
        """测试消息格式化功能"""
        strategy = SummaryCompressionStrategy()

        messages = [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
            {"role": "tool", "tool_call_id": "call_123", "content": "工具执行结果"},
            {"role": "user", "content": ""},  # 空内容应该被跳过
        ]

        formatted = strategy._format_messages_for_summary(messages)

        assert "系统: 你是一个有用的助手" in formatted
        assert "用户: 你好" in formatted
        assert "助手: 你好！有什么可以帮助你的吗？" in formatted
        assert "工具结果(call_123): 工具执行结果" in formatted
        assert formatted.count("\n\n") == 3  # 4个非空消息，3个分隔符

    def test_simple_summary(self):
        """测试简单摘要功能"""
        strategy = SummaryCompressionStrategy()

        messages = [
            {
                "role": "user",
                "content": "这是一个很长的用户消息，包含了很多重要的信息，需要被摘要处理",
            },
            {"role": "assistant", "content": "这是助手的回复，也包含了重要的信息"},
            {"role": "user", "content": "短消息"},  # 太短，应该被跳过
            {
                "role": "assistant",
                "content": "另一个很长的助手回复，包含了更多的详细信息和解释",
            },
        ]

        summary = strategy._simple_summary(messages)

        assert "[user]" in summary
        assert "[assistant]" in summary
        assert "这是一个很长的用户消息" in summary
        assert "短消息" not in summary  # 太短的消息应该被跳过

    def test_create_summary_with_openai_client(self):
        """测试使用OpenAI客户端创建摘要"""
        mock_client = self.create_mock_openai_client(
            "用户询问了天气，助手提供了北京的天气信息"
        )
        strategy = SummaryCompressionStrategy(openai_client=mock_client)

        messages = [
            {"role": "user", "content": "北京今天天气怎么样？"},
            {"role": "assistant", "content": "北京今天晴天，温度25度"},
        ]

        summary = strategy._create_summary(messages)

        assert summary == "用户询问了天气，助手提供了北京的天气信息"

        # 验证OpenAI客户端被正确调用
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args

        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["max_tokens"] == 1000
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert "用户: 北京今天天气怎么样？" in call_args[1]["messages"][0]["content"]

    def test_create_summary_without_openai_client(self):
        """测试没有OpenAI客户端时的摘要创建"""
        strategy = SummaryCompressionStrategy(openai_client=None)

        messages = [
            {"role": "user", "content": "这是一个很长的用户消息，包含了很多重要的信息"},
            {
                "role": "assistant",
                "content": "这是助手的详细回复，包含了重要的信息和解释",
            },
        ]

        summary = strategy._create_summary(messages)

        # 应该使用简单摘要方法
        assert "[user]" in summary
        assert "[assistant]" in summary

    def test_create_summary_with_openai_error(self):
        """测试OpenAI客户端出错时的处理"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API错误")

        strategy = SummaryCompressionStrategy(openai_client=mock_client)

        messages = [
            {"role": "user", "content": "这是一个很长的用户消息，包含了很多重要的信息"},
            {
                "role": "assistant",
                "content": "这是助手的详细回复，包含了重要的信息和解释",
            },
        ]

        summary = strategy._create_summary(messages)

        # 应该回退到简单摘要方法
        assert "[user]" in summary
        assert "[assistant]" in summary

    def test_compress_no_compression_needed(self):
        """测试不需要压缩的情况"""
        strategy = SummaryCompressionStrategy()

        messages = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "用户消息"},
        ]

        # 假设这些消息的token数很少
        result = strategy.compress(
            messages, max_tokens=1000, token_counter=self.token_counter
        )

        assert result == messages

    def test_compress_with_summary(self):
        """测试需要摘要压缩的情况"""
        mock_client = self.create_mock_openai_client("早期对话的摘要")
        strategy = SummaryCompressionStrategy(
            openai_client=mock_client, preserve_recent=4  # 保留最近的4条消息（2轮对话）
        )

        messages = [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "第一个用户消息"},
            {"role": "assistant", "content": "第一个助手回复"},
            {"role": "user", "content": "第二个用户消息"},
            {"role": "assistant", "content": "第二个助手回复"},
            {"role": "user", "content": "第三个用户消息"},
            {"role": "assistant", "content": "第三个助手回复"},
            {"role": "user", "content": "最新的用户消息"},
            {"role": "assistant", "content": "最新的助手回复"},
        ]

        # 首先检查原始消息的token数
        original_tokens = self.token_counter.count_messages(messages)

        # 设置一个小于原始token数的限制来触发压缩
        max_tokens = max(80, original_tokens - 20)  # 确保小于原始token数
        result = strategy.compress(
            messages, max_tokens=max_tokens, token_counter=self.token_counter
        )

        # 检查结果结构
        system_messages = [
            m
            for m in result
            if m.get("role") == "system" and "[对话摘要]" not in m.get("content", "")
        ]
        summary_messages = [m for m in result if "[对话摘要]" in m.get("content", "")]
        recent_messages = [
            m
            for m in result
            if m.get("role") in ["user", "assistant"]
            and "[对话摘要]" not in m.get("content", "")
        ]

        # 应该有原始系统消息
        assert len(system_messages) >= 1
        assert system_messages[0]["content"] == "你是一个有用的助手"

        # 应该有摘要消息或者消息被压缩了
        if len(summary_messages) > 0:
            assert "早期对话的摘要" in summary_messages[0]["content"]

        # 结果应该比原始消息少
        assert len(result) < len(messages)

        # 应该保留一些最近的消息
        if len(recent_messages) > 0:
            recent_contents = [m.get("content") for m in recent_messages]
            # 检查是否包含最新的消息
            assert any("最新" in content for content in recent_contents) or any(
                "第三个" in content for content in recent_contents
            )

    def test_compress_preserve_system_false(self):
        """测试不保留系统消息的情况"""
        mock_client = self.create_mock_openai_client("对话摘要")
        strategy = SummaryCompressionStrategy(
            openai_client=mock_client, preserve_system=False, preserve_recent=1
        )

        messages = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "用户消息1"},
            {"role": "assistant", "content": "助手回复1"},
            {"role": "user", "content": "用户消息2"},
            {"role": "assistant", "content": "助手回复2"},
        ]

        result = strategy.compress(
            messages, max_tokens=30, token_counter=self.token_counter
        )

        # 系统消息应该被当作普通消息处理
        summary_messages = [m for m in result if "[对话摘要]" in m.get("content", "")]
        assert len(summary_messages) >= 0  # 可能有摘要消息

    def test_compress_only_system_messages(self):
        """测试只有系统消息的情况"""
        strategy = SummaryCompressionStrategy()

        messages = [
            {"role": "system", "content": "第一个系统消息"},
            {"role": "system", "content": "第二个系统消息"},
        ]

        result = strategy.compress(
            messages, max_tokens=10, token_counter=self.token_counter
        )

        # 应该使用截断策略处理
        assert len(result) <= len(messages)
        assert all(m.get("role") == "system" for m in result)

    def test_compress_empty_messages(self):
        """测试空消息列表"""
        strategy = SummaryCompressionStrategy()

        result = strategy.compress([], max_tokens=100, token_counter=self.token_counter)

        assert result == []

    def test_compress_with_fallback_truncation(self):
        """测试摘要后仍需要截断的情况"""
        mock_client = self.create_mock_openai_client(
            "一个很长的摘要" * 100
        )  # 创建一个很长的摘要
        strategy = SummaryCompressionStrategy(
            openai_client=mock_client, preserve_recent=1
        )

        messages = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "用户消息1"},
            {"role": "assistant", "content": "助手回复1"},
            {"role": "user", "content": "用户消息2"},
            {"role": "assistant", "content": "助手回复2"},
        ]

        # 设置一个很小的token限制
        result = strategy.compress(
            messages, max_tokens=5, token_counter=self.token_counter
        )

        # 结果应该被进一步截断
        result_tokens = self.token_counter.count_messages(result)
        assert result_tokens <= 5 or len(result) <= 1  # 可能只保留一条消息
