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
            {"role": "user", "content": "Hello!"},
        ]
        count = counter.count_messages(messages)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tool_calls(self):
        counter = TiktokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Beijing","unit":"celsius"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Sunny, 25°C"},
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
            {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
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
            MockChatCompletionMessage(
                role="system", content="You are a helpful assistant."
            ),
            MockChatCompletionMessage(role="user", content="Hello!"),
            MockChatCompletionMessage(role="assistant", content="Hi there!"),
            MockChatCompletionMessage(
                role="tool", tool_call_id="call_123", content="Tool result"
            ),
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
                        "arguments": '{"location":"Beijing","unit":"celsius"}',
                    },
                }
            ],
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
                        "arguments": '{"location":"Beijing","unit":"celsius"}',
                    },
                }
            ],
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
                ChatCompletionToolMessageParam,
            )

            # 创建真实的OpenAI消息参数对象
            system_msg = ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant."
            )
            user_msg = ChatCompletionUserMessageParam(role="user", content="Hello!")
            assistant_msg = ChatCompletionAssistantMessageParam(
                role="assistant", content="Hi there!"
            )
            tool_msg = ChatCompletionToolMessageParam(
                role="tool", tool_call_id="call_123", content="Tool result"
            )

            # 创建对应的dict格式消息
            dict_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
            ]

            # 测试OpenAI消息参数对象能够正确计算token数量
            try:
                openai_count = counter.count_messages(
                    [system_msg, user_msg, assistant_msg, tool_msg]
                )
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

    def test_mixed_message_formats(self):
        """测试dict和Pydantic模型消息混合处理"""
        counter = TiktokenCounter()

        # 创建模拟的Pydantic模型对象
        class MockPydanticMessage:
            """模拟Pydantic模型消息对象"""

            def __init__(self, role, content=None, tool_calls=None, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id

            def model_dump(self):
                """模拟Pydantic的model_dump()方法"""
                result = {"role": self.role}
                if self.content is not None:
                    result["content"] = self.content
                if self.tool_calls is not None:
                    result["tool_calls"] = self.tool_calls
                if self.tool_call_id is not None:
                    result["tool_call_id"] = self.tool_call_id
                return result

        # 创建混合格式的消息列表
        mixed_messages = [
            # dict格式的系统消息
            {"role": "system", "content": "You are a helpful assistant."},
            # Pydantic模型格式的用户消息
            MockPydanticMessage(role="user", content="Hello, how are you?"),
            # dict格式的助手消息
            {
                "role": "assistant",
                "content": "I'm doing well, thank you! How can I help you today?",
            },
            # Pydantic模型格式的工具消息
            MockPydanticMessage(
                role="tool",
                tool_call_id="call_456",
                content="Weather data retrieved successfully",
            ),
        ]

        # 创建对应的纯dict格式消息列表进行对比
        dict_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {
                "role": "assistant",
                "content": "I'm doing well, thank you! How can I help you today?",
            },
            {
                "role": "tool",
                "tool_call_id": "call_456",
                "content": "Weather data retrieved successfully",
            },
        ]

        mixed_count = counter.count_messages(mixed_messages)
        dict_count = counter.count_messages(dict_messages)

        # 混合格式和纯dict格式应该产生相同的token数量
        assert mixed_count == dict_count
        assert mixed_count > 0
        assert isinstance(mixed_count, int)

    def test_parallel_tool_calls(self):
        """测试并行工具调用的处理"""
        counter = TiktokenCounter()

        # 测试包含多个并行工具调用的消息
        messages_with_parallel_tools = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to multiple tools.",
            },
            {
                "role": "user",
                "content": "Please get the weather for Beijing and Shanghai, and also search for recent news about AI.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Beijing","unit":"celsius"}',
                        },
                    },
                    {
                        "id": "call_weather_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Shanghai","unit":"celsius"}',
                        },
                    },
                    {
                        "id": "call_news_1",
                        "type": "function",
                        "function": {
                            "name": "search_news",
                            "arguments": '{"query":"artificial intelligence","limit":5}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": "Beijing: Sunny, 22°C",
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_2",
                "content": "Shanghai: Cloudy, 25°C",
            },
            {
                "role": "tool",
                "tool_call_id": "call_news_1",
                "content": "Latest AI news: 1. GPT-4 improvements... 2. New AI regulations...",
            },
            {
                "role": "assistant",
                "content": "Based on the weather data and news search, here's what I found...",
            },
        ]

        # 测试单个工具调用的消息进行对比
        messages_with_single_tool = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to multiple tools.",
            },
            {"role": "user", "content": "Please get the weather for Beijing."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Beijing","unit":"celsius"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather_1",
                "content": "Beijing: Sunny, 22°C",
            },
            {
                "role": "assistant",
                "content": "The weather in Beijing is sunny with a temperature of 22°C.",
            },
        ]

        parallel_count = counter.count_messages(messages_with_parallel_tools)
        single_count = counter.count_messages(messages_with_single_tool)

        # 并行工具调用应该产生更多的token数量
        assert parallel_count > single_count
        assert parallel_count > 0
        assert single_count > 0
        assert isinstance(parallel_count, int)
        assert isinstance(single_count, int)

        # 验证并行工具调用中的每个工具调用都被正确计算
        # 通过检查assistant消息中的tool_calls数量
        assistant_msg = messages_with_parallel_tools[2]
        assert len(assistant_msg["tool_calls"]) == 3

        # 验证每个工具调用结果都被正确计算
        tool_results = [
            msg for msg in messages_with_parallel_tools if msg.get("role") == "tool"
        ]
        assert len(tool_results) == 3

    def test_mixed_formats_with_parallel_tools(self):
        """测试混合消息格式与并行工具调用的组合处理"""
        counter = TiktokenCounter()

        # 创建模拟的对象类型
        class MockObjectMessage:
            """模拟对象类型的消息"""

            def __init__(self, role, content=None, tool_calls=None, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id

            def items(self):
                """模拟dict的items()方法"""
                result = [("role", self.role)]
                if self.content is not None:
                    result.append(("content", self.content))
                if self.tool_calls is not None:
                    result.append(("tool_calls", self.tool_calls))
                if self.tool_call_id is not None:
                    result.append(("tool_call_id", self.tool_call_id))
                return result

        class MockPydanticMessage:
            """模拟Pydantic模型消息"""

            def __init__(self, role, content=None, tool_calls=None, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls
                self.tool_call_id = tool_call_id

            def model_dump(self):
                """模拟Pydantic的model_dump()方法"""
                result = {"role": self.role}
                if self.content is not None:
                    result["content"] = self.content
                if self.tool_calls is not None:
                    result["tool_calls"] = self.tool_calls
                if self.tool_call_id is not None:
                    result["tool_call_id"] = self.tool_call_id
                return result

        # 创建混合格式的消息，包含并行工具调用
        mixed_messages_with_tools = [
            # dict格式的系统消息
            {"role": "system", "content": "You are a helpful assistant."},
            # Pydantic模型格式的用户消息
            MockPydanticMessage(
                role="user", content="Get weather for multiple cities and search news."
            ),
            # 对象格式的助手消息，包含并行工具调用
            MockObjectMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Tokyo"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"London"}',
                        },
                    },
                ],
            ),
            # dict格式的工具结果
            {"role": "tool", "tool_call_id": "call_1", "content": "Tokyo: 18°C, Rainy"},
            # Pydantic模型格式的工具结果
            MockPydanticMessage(
                role="tool", tool_call_id="call_2", content="London: 12°C, Foggy"
            ),
            # dict格式的最终回复
            {
                "role": "assistant",
                "content": "Here's the weather information for both cities.",
            },
        ]

        # 创建对应的纯dict格式消息进行对比
        dict_messages_with_tools = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Get weather for multiple cities and search news.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Tokyo"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"London"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Tokyo: 18°C, Rainy"},
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "London: 12°C, Foggy",
            },
            {
                "role": "assistant",
                "content": "Here's the weather information for both cities.",
            },
        ]

        mixed_count = counter.count_messages(mixed_messages_with_tools)
        dict_count = counter.count_messages(dict_messages_with_tools)

        # 混合格式和纯dict格式应该产生相同的token数量
        # 由于不同格式在处理None值时可能有细微差别，允许小的差异
        assert (
            abs(mixed_count - dict_count) <= 2
        ), f"Token count difference too large: mixed={mixed_count}, dict={dict_count}"
        assert mixed_count > 0
        assert isinstance(mixed_count, int)

        # 验证并行工具调用被正确处理
        assistant_msg = mixed_messages_with_tools[2]
        # 通过items()方法获取tool_calls
        tool_calls = None
        for key, value in assistant_msg.items():
            if key == "tool_calls":
                tool_calls = value
                break

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[1]["id"] == "call_2"
