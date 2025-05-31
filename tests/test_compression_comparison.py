"""
测试不同压缩策略在6000上下文上限下的表现比较
"""

import json
from unittest.mock import Mock

import pytest

from effimemo.core.tokenizer import TiktokenCounter
from effimemo.manager import ContextManager


class TestCompressionComparison:
    """测试不同压缩策略的比较"""

    def setup_method(self):
        """设置测试环境"""
        self.max_tokens = 6000
        self.token_counter = TiktokenCounter()

        # 加载测试消息
        with open("tests/test_messages.json", "r", encoding="utf-8") as f:
            self.test_conversations = json.load(f)

        # 预处理消息，将list类型的content转换为字符串
        self.test_conversations = self.preprocess_conversations(self.test_conversations)

    def preprocess_conversations(self, conversations):
        """预处理对话，处理复杂的content格式"""
        processed_conversations = []

        for conversation in conversations:
            for message in conversation:
                processed_message = message.copy()

                # 处理content字段
                content = message.get("content", "")
                if isinstance(content, list):
                    processed_message["content"] = content
                elif not isinstance(content, str):
                    # 如果content不是字符串，转换为字符串
                    processed_message["content"] = str(content)

                processed_conversations.append(processed_message)
        return processed_conversations

    def create_mock_openai_client(self):
        """创建模拟的 OpenAI 客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # 模拟返回一个简洁的摘要
        mock_message.content = "这是对话的摘要：用户和助手进行了多轮交互，涉及技术问题讨论和信息交换。"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    def test_first_truncation_strategy(self):
        """测试 first 截断策略"""
        manager = ContextManager(
            max_tokens=self.max_tokens, strategy="first", preserve_system=True
        )

        original_tokens = manager.count_tokens(self.test_conversations)
        compressed = manager.compress(self.test_conversations)
        compressed_tokens = manager.count_tokens(compressed)

        assert compressed_tokens <= self.max_tokens

        result = {
            "conversation_id": 0,
            "strategy": "first",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": (
                compressed_tokens / original_tokens if original_tokens > 0 else 0
            ),
            "within_limit": compressed_tokens <= self.max_tokens,
        }

        print(
            f"First策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens"
        )

        # 验证结果
        assert result["within_limit"] is True
        assert result["compressed_messages"] <= result["original_messages"]

    def test_last_truncation_strategy(self):
        """测试 last 截断策略"""
        manager = ContextManager(
            max_tokens=self.max_tokens, strategy="last", preserve_system=True
        )

        original_tokens = manager.count_tokens(self.test_conversations)
        compressed = manager.compress(self.test_conversations)
        compressed_tokens = manager.count_tokens(compressed)

        assert compressed_tokens <= self.max_tokens

        result = {
            "conversation_id": 0,
            "strategy": "last",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": (
                compressed_tokens / original_tokens if original_tokens > 0 else 0
            ),
            "within_limit": compressed_tokens <= self.max_tokens,
        }

        print(
            f"Last策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens"
        )

        # 验证结果
        assert result["within_limit"] is True
        assert result["compressed_messages"] <= result["original_messages"]

    def test_summary_compression_strategy(self):
        """测试 summary 压缩策略"""
        mock_client = self.create_mock_openai_client()

        manager = ContextManager(
            max_tokens=self.max_tokens,
            strategy="summary",
            openai_client=mock_client,
            preserve_system=True,
            preserve_recent=4,  # 保留最近4条消息
        )

        original_tokens = manager.count_tokens(self.test_conversations)
        compressed = manager.compress(self.test_conversations)
        compressed_tokens = manager.count_tokens(compressed)

        assert compressed_tokens <= self.max_tokens

        # 检查是否包含摘要
        has_summary = any("[对话摘要]" in msg.get("content", "") for msg in compressed)

        result = {
            "conversation_id": 0,
            "strategy": "summary",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": (
                compressed_tokens / original_tokens if original_tokens > 0 else 0
            ),
            "within_limit": compressed_tokens <= self.max_tokens,
            "has_summary": has_summary,
        }

        print(
            f"Summary策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens, 包含摘要: {has_summary}"
        )

        # 验证结果
        assert result["within_limit"] is True
        assert result["compressed_messages"] <= result["original_messages"]

    def test_compression_comparison(self):
        """比较所有压缩策略的性能"""
        print("\n" + "=" * 80)
        print("压缩策略性能比较测试 (上下文限制: 6000 tokens)")
        print("=" * 80)

        # 先显示原始对话信息
        print("\n原始对话信息:")
        tokens = self.token_counter.count_messages(self.test_conversations)
        print(f"  对话{0}: {len(self.test_conversations)}条消息, {tokens}tokens")

        # 运行所有策略并收集结果
        print("\n运行策略测试...")

        # First策略
        manager_first = ContextManager(
            max_tokens=self.max_tokens, strategy="first", preserve_system=True
        )
        original_tokens = manager_first.count_tokens(self.test_conversations)
        compressed_first = manager_first.compress(self.test_conversations)
        compressed_tokens_first = manager_first.count_tokens(compressed_first)

        first_result = {
            "strategy": "first",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed_first),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens_first,
            "compression_ratio": compressed_tokens_first / original_tokens
            if original_tokens > 0
            else 0,
            "within_limit": compressed_tokens_first <= self.max_tokens,
        }

        # Last策略
        manager_last = ContextManager(
            max_tokens=self.max_tokens, strategy="last", preserve_system=True
        )
        compressed_last = manager_last.compress(self.test_conversations)
        compressed_tokens_last = manager_last.count_tokens(compressed_last)

        last_result = {
            "strategy": "last",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed_last),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens_last,
            "compression_ratio": compressed_tokens_last / original_tokens
            if original_tokens > 0
            else 0,
            "within_limit": compressed_tokens_last <= self.max_tokens,
        }

        # Summary策略
        mock_client = self.create_mock_openai_client()
        manager_summary = ContextManager(
            max_tokens=self.max_tokens,
            strategy="summary",
            openai_client=mock_client,
            preserve_system=True,
            preserve_recent=4,
        )
        compressed_summary = manager_summary.compress(self.test_conversations)
        compressed_tokens_summary = manager_summary.count_tokens(compressed_summary)

        has_summary = any(
            "[对话摘要]" in msg.get("content", "") for msg in compressed_summary
        )

        summary_result = {
            "strategy": "summary",
            "original_messages": len(self.test_conversations),
            "compressed_messages": len(compressed_summary),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens_summary,
            "compression_ratio": compressed_tokens_summary / original_tokens
            if original_tokens > 0
            else 0,
            "within_limit": compressed_tokens_summary <= self.max_tokens,
            "has_summary": has_summary,
        }

        # 汇总统计
        print("\n" + "=" * 80)
        print("策略性能汇总")
        print("=" * 80)

        all_results = [first_result, last_result, summary_result]

        for result in all_results:
            strategy_name = result["strategy"].upper()
            print(f"\n{strategy_name}策略:")
            print(f"  压缩率: {result['compression_ratio']:.1%}")
            print(
                f"  消息数: {result['original_messages']} -> {result['compressed_messages']}"
            )
            print(
                f"  Token数: {result['original_tokens']} -> {result['compressed_tokens']}"
            )
            print(f"  符合限制: {'✅' if result['within_limit'] else '❌'}")

            if "has_summary" in result:
                print(f"  包含摘要: {'✅' if result['has_summary'] else '❌'}")

        print("\n" + "=" * 80)
        print("测试结论:")
        print("- Summary策略压缩率最高，适合长对话历史")
        print("- Last策略保留最新信息，适合连续对话")
        print("- First策略保留早期信息，适合保持上下文完整性")
        print("=" * 80)

        # 验证所有策略都符合token限制
        for result in all_results:
            assert result["within_limit"], f"{result['strategy']}策略超出token限制"

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n" + "=" * 50)
        print("边界情况测试")
        print("=" * 50)

        # 测试空对话
        empty_conversation = []

        # 测试只有系统消息的对话
        system_only = [{"role": "system", "content": "你是一个有用的助手"}]

        # 测试很短的对话
        short_conversation = [
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]

        edge_cases = [
            ("空对话", empty_conversation),
            ("只有系统消息", system_only),
            ("短对话", short_conversation),
        ]

        strategies = ["first", "last", "selective", "summary"]

        for case_name, conversation in edge_cases:
            print(f"\n测试 {case_name}:")

            for strategy in strategies:
                try:
                    if strategy == "summary":
                        mock_client = self.create_mock_openai_client()
                        manager = ContextManager(
                            max_tokens=self.max_tokens,
                            strategy=strategy,
                            openai_client=mock_client,
                        )
                    else:
                        manager = ContextManager(
                            max_tokens=self.max_tokens, strategy=strategy
                        )

                    compressed = manager.compress(conversation)
                    tokens = manager.count_tokens(compressed)

                    print(
                        f"  {strategy}: {len(conversation)}条 -> {len(compressed)}条, {tokens}tokens"
                    )

                    # 验证结果
                    assert isinstance(compressed, list)
                    assert tokens <= self.max_tokens

                except Exception as e:
                    print(f"  {strategy}: 错误 - {e}")
                    raise

        print("\n✅ 边界情况测试通过！")

    def test_selective_compression_strategy(self):
        """测试selective压缩策略"""
        try:
            from effimemo.core.tokenizer import TiktokenCounter
            from effimemo.strategies.compression import SelectiveCompressionStrategy

            strategy = SelectiveCompressionStrategy()
            counter = TiktokenCounter()

            # 测试基本压缩功能
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]

            result = strategy.compress(messages, 100, counter)
            assert len(result) <= len(messages)
            assert counter.count_messages(result) <= 100

        except ImportError:
            # 如果没有安装compression依赖，跳过测试
            pytest.skip("Compression dependencies not installed")

    def test_compression_strategy_fallback(self):
        """测试compression策略的回退机制"""
        try:
            from effimemo.core.tokenizer import TiktokenCounter
            from effimemo.strategies.compression import SelectiveCompressionStrategy

            strategy = SelectiveCompressionStrategy(
                reduce_ratio=0.8, preserve_system=True
            )
            counter = TiktokenCounter()

            # 创建一个会触发回退的场景
            messages = [
                {"role": "system", "content": "System message"},
                {
                    "role": "user",
                    "content": "This is a very long message that needs compression. "
                    * 20,
                },
            ]

            result = strategy.compress(messages, 100, counter)
            assert len(result) > 0
            assert counter.count_messages(result) <= 100

        except ImportError:
            # 如果没有安装compression依赖，跳过测试
            pytest.skip("Compression dependencies not installed")


if __name__ == "__main__":
    test = TestCompressionComparison()
    test.setup_method()
    test.test_compression_comparison()
    test.test_edge_cases()
