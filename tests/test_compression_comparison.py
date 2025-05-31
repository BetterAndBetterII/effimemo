"""
测试不同压缩策略在6000上下文上限下的表现比较
"""

import json
import pytest
from unittest.mock import Mock
from effimemo.manager import ContextManager
from effimemo.core.tokenizer import TiktokenCounter


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
        mock_message.content = (
            "这是对话的摘要：用户和助手进行了多轮交互，涉及技术问题讨论和信息交换。"
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    def test_first_truncation_strategy(self):
        """测试 first 截断策略"""
        manager = ContextManager(
            max_tokens=self.max_tokens, strategy="first", preserve_system=True
        )

        results = []
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
        results.append(result)

        print(
            f"First策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens"
        )

        return results

    def test_last_truncation_strategy(self):
        """测试 last 截断策略"""
        manager = ContextManager(
            max_tokens=self.max_tokens, strategy="last", preserve_system=True
        )

        results = []
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
        results.append(result)

        print(
            f"Last策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens"
        )

        return results

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

        results = []
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
        results.append(result)

        print(
            f"Summary策略 - 对话{0}: {len(self.test_conversations)}条消息 -> {len(compressed)}条消息, "
            f"{original_tokens}tokens -> {compressed_tokens}tokens, 包含摘要: {has_summary}"
        )

        return results

    def test_compression_comparison(self):
        """比较所有压缩策略的性能"""
        print("\n" + "=" * 80)
        print("压缩策略性能比较测试 (上下文限制: 6000 tokens)")
        print("=" * 80)

        # 先显示原始对话信息
        print("\n原始对话信息:")
        tokens = self.token_counter.count_messages(self.test_conversations)
        print(f"  对话{0}: {len(self.test_conversations)}条消息, {tokens}tokens")

        # 运行所有策略
        first_results = self.test_first_truncation_strategy()
        print()
        last_results = self.test_last_truncation_strategy()
        print()
        summary_results = self.test_summary_compression_strategy()

        # 汇总统计
        print("\n" + "=" * 80)
        print("策略性能汇总")
        print("=" * 80)

        all_results = {
            "first": first_results,
            "last": last_results,
            "summary": summary_results,
        }

        for strategy_name, results in all_results.items():
            if not results:
                continue

            total_conversations = len(results)
            avg_compression_ratio = (
                sum(r["compression_ratio"] for r in results) / total_conversations
            )
            within_limit_count = sum(1 for r in results if r["within_limit"])
            within_limit_rate = within_limit_count / total_conversations * 100

            avg_original_messages = (
                sum(r["original_messages"] for r in results) / total_conversations
            )
            avg_compressed_messages = (
                sum(r["compressed_messages"] for r in results) / total_conversations
            )

            avg_original_tokens = (
                sum(r["original_tokens"] for r in results) / total_conversations
            )
            avg_compressed_tokens = (
                sum(r["compressed_tokens"] for r in results) / total_conversations
            )

            print(f"\n{strategy_name.upper()} 策略:")
            print(f"  平均压缩比: {avg_compression_ratio:.3f}")
            print(
                f"  符合限制率: {within_limit_rate:.1f}% ({within_limit_count}/{total_conversations})"
            )
            print(
                f"  平均消息数: {avg_original_messages:.1f} -> {avg_compressed_messages:.1f}"
            )
            print(
                f"  平均token数: {avg_original_tokens:.0f} -> {avg_compressed_tokens:.0f}"
            )

            if strategy_name == "summary":
                has_summary_count = sum(
                    1 for r in results if r.get("has_summary", False)
                )
                print(
                    f"  生成摘要率: {has_summary_count/total_conversations*100:.1f}% ({has_summary_count}/{total_conversations})"
                )

        # 验证所有策略都能将结果控制在限制范围内
        for strategy_name, results in all_results.items():
            for result in results:
                assert result[
                    "within_limit"
                ], f"{strategy_name}策略在对话{result['conversation_id']}中超出了token限制"

        print("\n✅ 所有压缩策略测试通过！")

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


if __name__ == "__main__":
    test = TestCompressionComparison()
    test.setup_method()
    test.test_compression_comparison()
    test.test_edge_cases()
