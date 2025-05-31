"""
测试 ContextManager 对 summary 策略的支持
"""

from unittest.mock import Mock
from effimemo.manager import ContextManager


def test_manager_with_summary_strategy():
    """测试使用 summary 策略初始化 ContextManager"""

    # 创建 mock OpenAI 客户端
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()

    mock_message.content = "这是一个测试摘要"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    # 使用 summary 策略初始化 manager
    manager = ContextManager(
        max_tokens=1000,
        strategy="summary",
        openai_client=mock_client,
        summary_model="gpt-4",
        preserve_recent=2,
        preserve_system=True,
    )

    # 验证策略类型
    from effimemo.strategies.summary import SummaryCompressionStrategy

    assert isinstance(manager.strategy, SummaryCompressionStrategy)

    # 验证策略参数
    assert manager.strategy.openai_client == mock_client
    assert manager.strategy.model == "gpt-4"
    assert manager.strategy.preserve_recent == 2
    assert manager.strategy.preserve_system == True

    print("✅ ContextManager 成功支持 summary 策略!")


def test_manager_summary_compression():
    """测试 manager 使用 summary 策略进行压缩"""

    # 创建 mock OpenAI 客户端
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()

    mock_message.content = "用户询问了天气，助手提供了详细的天气信息"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    # 初始化 manager
    manager = ContextManager(
        max_tokens=100,  # 设置较小的限制来触发压缩
        strategy="summary",
        openai_client=mock_client,
        preserve_recent=2,
    )

    # 测试消息
    messages = [
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "今天是晴天，温度适宜，非常适合外出活动。"},
        {"role": "user", "content": "明天呢？"},
        {"role": "assistant", "content": "明天可能会有小雨，建议带伞出门。"},
        {"role": "user", "content": "谢谢你的建议"},
        {"role": "assistant", "content": "不客气，有其他问题随时问我！"},
    ]

    # 执行压缩
    compressed = manager.compress(messages)

    # 验证结果
    assert len(compressed) < len(messages)  # 应该被压缩了

    # 检查是否包含摘要
    summary_found = any("[对话摘要]" in msg.get("content", "") for msg in compressed)

    print(f"原始消息数量: {len(messages)}")
    print(f"压缩后消息数量: {len(compressed)}")
    print(f"包含摘要: {summary_found}")
    print("✅ ContextManager summary 压缩测试通过!")


if __name__ == "__main__":
    test_manager_with_summary_strategy()
    test_manager_summary_compression()
    print("🎉 所有测试通过!")
