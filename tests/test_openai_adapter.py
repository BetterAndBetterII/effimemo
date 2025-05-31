import pytest
from effimemo.adapters.openai import OpenAIAdapter


class TestOpenAIAdapter:
    def test_is_valid_message(self):
        # 有效消息
        assert OpenAIAdapter.is_valid_message({"role": "user", "content": "Hello"})
        assert OpenAIAdapter.is_valid_message(
            {"role": "system", "content": "You are an assistant"}
        )
        assert OpenAIAdapter.is_valid_message(
            {"role": "assistant", "content": "Hi there"}
        )
        assert OpenAIAdapter.is_valid_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "123", "function": {"name": "test", "arguments": "{}"}}
                ],
            }
        )
        assert OpenAIAdapter.is_valid_message(
            {"role": "tool", "tool_call_id": "123", "content": "Result"}
        )

        # 无效消息
        assert not OpenAIAdapter.is_valid_message({})
        assert not OpenAIAdapter.is_valid_message({"content": "No role"})
        assert not OpenAIAdapter.is_valid_message(
            {"role": "invalid", "content": "Invalid role"}
        )
        assert not OpenAIAdapter.is_valid_message({"role": "user"})  # 缺少content
        assert not OpenAIAdapter.is_valid_message(
            {"role": "tool", "content": "Missing tool_call_id"}
        )

    def test_validate_messages(self):
        valid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        assert OpenAIAdapter.validate_messages(valid_messages)

        invalid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "invalid", "content": "This role is invalid"},
        ]
        assert not OpenAIAdapter.validate_messages(invalid_messages)

        assert not OpenAIAdapter.validate_messages("not a list")

    def test_extract_tool_calls(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"New York"}',
                        },
                    },
                    {
                        "id": "call2",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone":"EST"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "Sunny, 25°C"},
        ]

        tool_calls = OpenAIAdapter.extract_tool_calls(messages)
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "call1"
        assert tool_calls[1]["id"] == "call2"

    def test_extract_tool_results(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"New York"}',
                        },
                    },
                    {
                        "id": "call2",
                        "function": {
                            "name": "get_time",
                            "arguments": '{"timezone":"EST"}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "Sunny, 25°C"},
            {"role": "tool", "tool_call_id": "call2", "content": "10:30 AM"},
        ]

        tool_results = OpenAIAdapter.extract_tool_results(messages)
        assert len(tool_results) == 2
        assert tool_results[0]["tool_call_id"] == "call1"
        assert tool_results[1]["tool_call_id"] == "call2"
