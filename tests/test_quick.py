def test_summary_strategy():
    """测试摘要策略（模拟）"""
    print("\n🧪 测试摘要策略...")
    
    # 创建模拟的OpenAI客户端
    class MockOpenAIClient:
        class Chat:
            class Completions:
                def create(self, **kwargs):
                    class MockResponse:
                        class Choice:
                            class Message:
                                content = "这是对话的摘要：用户询问天气，助手回复无法提供实时信息。"
                            message = Message()
                        choices = [Choice()]
                    return MockResponse()
            completions = Completions()
        chat = Chat()
    
    mock_client = MockOpenAIClient()
    
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好！" * 50},  # 长消息
        {"role": "assistant", "content": "你好！我是你的AI助手。" * 50},
        {"role": "user", "content": "请告诉我今天的天气如何？" * 50},
        {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。" * 50},
    ]
    
    # 直接创建ContextManager并设置摘要策略
    from effimemo.manager import ContextManager
    from effimemo.strategies.summary import SummaryCompressionStrategy
    
    # 创建摘要策略实例
    summary_strategy = SummaryCompressionStrategy(
        openai_client=mock_client,
        preserve_recent=2
    )
    
    manager = ContextManager(
        max_tokens=500,
        strategy=summary_strategy,  # 直接传入策略实例
        preserve_system=True
    )
    
    original_tokens = manager.count_tokens(messages)
    print(f"  原始消息: {len(messages)}条, {original_tokens} tokens")
    
    compressed = manager.compress(messages)
    compressed_tokens = manager.count_tokens(compressed)
    print(f"  压缩后: {len(compressed)}条, {compressed_tokens} tokens")
    
    # 检查是否包含摘要
    has_summary = any("[对话摘要]" in str(msg.get("content", "")) for msg in compressed)
    print(f"  包含摘要: {'✅' if has_summary else '❌'}")
    
    assert compressed_tokens <= 500, "摘要策略未能控制在token限制内"
    print("  ✅ 摘要策略测试通过")