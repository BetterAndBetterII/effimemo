import pytest
from effimemo.core.tokenizer import TiktokenCounter, CachedTokenCounter

class TestTokenCounter:
    def test_count_empty_text(self):
        counter = TiktokenCounter()
        assert counter.count("") == 0
        
    def test_count_simple_text(self):
        counter = TiktokenCounter()
        text = "Hello, world!"
        count = counter.count(text)
        assert count > 0
        assert isinstance(count, int)
        
    def test_count_empty_messages(self):
        counter = TiktokenCounter()
        assert counter.count_messages([]) == 0
        
    def test_count_simple_messages(self):
        counter = TiktokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        count = counter.count_messages(messages)
        assert count > 0
        assert isinstance(count, int)
        
    def test_count_tool_calls(self):
        counter = TiktokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Beijing\",\"unit\":\"celsius\"}"
                    }
                }
            ]},
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Sunny, 25°C"}
        ]
        count = counter.count_messages(messages)
        assert count > 0
        assert isinstance(count, int)
        
    def test_count_openai_message_objects(self):
        """测试处理OpenAI库的ChatCompletionMessage对象和dict形式的消息"""
        counter = TiktokenCounter()
        
        # 首先测试dict形式的消息（标准格式）
        dict_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
        ]
        dict_count = counter.count_messages(dict_messages)
        
        # 测试模拟的ChatCompletionMessage对象
        # 由于我们可能没有安装openai库，我们创建模拟对象来测试
        class MockChatCompletionMessage:
            """模拟OpenAI的ChatCompletionMessage对象"""
            def __init__(self, role, content=None, tool_calls=None, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id
            
            def items(self):
                """模拟dict的items()方法，用于兼容现有的token计数逻辑"""
                result = [("role", self.role)]
                # 总是包含content字段，即使为None（与dict行为一致）
                result.append(("content", self.content))
                if self.tool_calls is not None:
                    result.append(("tool_calls", self.tool_calls))
                if self.tool_call_id is not None:
                    result.append(("tool_call_id", self.tool_call_id))
                return result
            
            def get(self, key, default=None):
                """模拟dict的get()方法"""
                return getattr(self, key, default)
        
        # 创建对应的ChatCompletionMessage对象
        object_messages = [
            MockChatCompletionMessage(role="system", content="You are a helpful assistant."),
            MockChatCompletionMessage(role="user", content="Hello!"),
            MockChatCompletionMessage(role="assistant", content="Hi there!"),
            MockChatCompletionMessage(role="tool", tool_call_id="call_123", content="Tool result")
        ]
        
        object_count = counter.count_messages(object_messages)
        
        # 两种格式应该产生相同的token数量
        assert object_count == dict_count
        assert object_count > 0
        assert isinstance(object_count, int)
        
        # 测试带有工具调用的消息
        tool_call_dict = {
            "role": "assistant", 
            "content": None, 
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Beijing\",\"unit\":\"celsius\"}"
                    }
                }
            ]
        }
        
        tool_call_object = MockChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Beijing\",\"unit\":\"celsius\"}"
                    }
                }
            ]
        )
        
        dict_tool_count = counter.count_messages([tool_call_dict])
        object_tool_count = counter.count_messages([tool_call_object])
        
        # 两种格式应该产生完全相同的token数量
        assert dict_tool_count == object_tool_count
        assert dict_tool_count > 0
        
    def test_count_real_openai_message_objects(self):
        """测试处理真实的OpenAI库ChatCompletionMessage对象（如果可用）"""
        counter = TiktokenCounter()
        
        try:
            # 尝试导入OpenAI库的消息参数类型
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam,
                ChatCompletionAssistantMessageParam,
                ChatCompletionToolMessageParam
            )
            
            # 创建真实的OpenAI消息参数对象
            system_msg = ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant.")
            user_msg = ChatCompletionUserMessageParam(role="user", content="Hello!")
            assistant_msg = ChatCompletionAssistantMessageParam(role="assistant", content="Hi there!")
            tool_msg = ChatCompletionToolMessageParam(role="tool", tool_call_id="call_123", content="Tool result")
            
            # 创建对应的dict格式消息
            dict_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
            ]
            
            # 测试OpenAI消息参数对象能够正确计算token数量
            try:
                openai_count = counter.count_messages([system_msg, user_msg, assistant_msg, tool_msg])
                dict_count = counter.count_messages(dict_messages)
                
                # 验证两种方式都能正确计算token数量
                assert openai_count > 0
                assert dict_count > 0
                assert isinstance(openai_count, int)
                assert isinstance(dict_count, int)
                
                # 由于OpenAI对象是Pydantic模型，应该通过model_dump()方法转换为dict
                # 所以token数量应该相等
                assert openai_count == dict_count
                
            except Exception as e:
                # 如果OpenAI对象结构不兼容，至少确保不会崩溃
                pytest.fail(f"处理真实OpenAI消息对象时出错: {e}")
                
        except ImportError:
            # 如果没有安装OpenAI库，跳过此测试
            pytest.skip("OpenAI库未安装，跳过真实OpenAI对象测试")
        
    def test_cached_counter(self):
        base_counter = TiktokenCounter()
        cached_counter = CachedTokenCounter(base_counter)
        
        text = "This is a test message that should be cached."
        
        # First call should calculate
        count1 = cached_counter.count(text)
        
        # Second call should use cache
        count2 = cached_counter.count(text)
        
        assert count1 == count2
        assert text in cached_counter.cache
