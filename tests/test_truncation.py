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
