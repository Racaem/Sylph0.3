from dataclasses import dataclass


@dataclass
class SylphError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class LexError(SylphError):
    """Raised when the lexer encounters an unexpected character."""


class ParseError(SylphError):
    """Raised when the parser encounters invalid syntax."""


class RuntimeError(SylphError):
    """Raised when the interpreter encounters an invalid state."""

