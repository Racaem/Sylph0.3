from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Any

from .errors import LexError


class TokenType(Enum):
    # Keywords
    OUT = auto()
    DEF = auto()
    LET = auto()
    IF = auto()
    RETURN = auto()
    END = auto()
    USE = auto()
    PUB = auto()
    
    # Types
    INT = auto()
    STRING = auto()
    BOOL = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    LESS = auto()
    EQUAL = auto()
    
    # Punctuation
    SEMICOLON = auto()
    COLON = auto()
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    COMMA = auto()
    DOT = auto()
    
    # Identifier
    IDENTIFIER = auto()
    
    # EOF
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str
    literal: Any = None
    line: int = 1
    column: int = 1


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.column = 1
        # 字典缓存关键词
        self.keywords = {
            "out": TokenType.OUT,
            "def": TokenType.DEF,
            "let": TokenType.LET,
            "if": TokenType.IF,
            "return": TokenType.RETURN,
            "end": TokenType.END,
            "true": (TokenType.BOOL, True),
            "false": (TokenType.BOOL, False),
            "use": TokenType.USE,
            "pub": TokenType.PUB
        }
        # Token对象池，缓存常用的Token对象
        self.token_pool: Dict[tuple, Token] = {}

    def scan_tokens(self) -> List[Token]:
        while not self._is_at_end():
            self.start = self.current
            self._scan_token()

        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    def _is_at_end(self) -> bool:
        return self.current >= len(self.source)

    def _advance(self) -> str:
        ch = self.source[self.current]
        self.current += 1
        self.column += 1
        return ch

    def _peek(self) -> str:
        if self._is_at_end():
            return "\0"
        return self.source[self.current]

    def _match(self, expected: str) -> bool:
        if self._is_at_end():
            return False
        if self.source[self.current] != expected:
            return False
        self.current += 1
        return True

    def _add_token(self, type_: TokenType, literal: Optional[Any] = None) -> None:
        text = self.source[self.start:self.current]
        # 计算列号，列号是token开始的位置相对于当前行的位置
        # 注意：我们需要计算token开始时的列号，即self.start相对于当前行的位置
        # 简化处理：直接使用当前行的列号，即self.column
        column = self.column
        if column < 1:
            column = 1
        # 使用Token的属性作为键，在对象池中查找或创建Token对象
        token_key = (type_, text, literal, self.line, column)
        if token_key not in self.token_pool:
            # 如果对象池中没有，创建新的Token对象并添加到对象池中
            self.token_pool[token_key] = Token(type_, text, literal, self.line, column)
        # 从对象池中获取Token对象并添加到tokens列表中
        self.tokens.append(self.token_pool[token_key])

    def _scan_token(self) -> None:
        c = self._advance()

        if c in (" ", "\r", "\t"):
            self.column += 1
            return
        if c == "\n":
            self.line += 1
            self.column = 1
            return

        if c == ";":
            self._add_token(TokenType.SEMICOLON)
            return
        if c == ":":
            self._add_token(TokenType.COLON)
            return
        if c == "(":
            self._add_token(TokenType.LEFT_PAREN)
            return
        if c == ")":
            self._add_token(TokenType.RIGHT_PAREN)
            return
        if c == "[":
            self._add_token(TokenType.LEFT_BRACKET)
            return
        if c == "]":
            self._add_token(TokenType.RIGHT_BRACKET)
            return
        if c == ",":
            self._add_token(TokenType.COMMA)
            return
        if c == ".":
            self._add_token(TokenType.DOT)
            return
        if c == "+":
            self._add_token(TokenType.PLUS)
            return
        if c == "-":
            self._add_token(TokenType.MINUS)
            return
        if c == "*":
            self._add_token(TokenType.MULTIPLY)
            return
        if c == "<":
            self._add_token(TokenType.LESS)
            return
        if c == "=":
            self._add_token(TokenType.EQUAL)
            return

        if c == '"':
            self._string()
            return

        if c.isdigit():
            self._number(c)
            return

        # Comments starting with // until end of line (convenience)
        if c == "/" and self._match("/"):
            while self._peek() != "\n" and not self._is_at_end():
                self._advance()
            return

        if c.isalpha():
            self._identifier_or_keyword(c)
            return

        raise LexError(f"LexError: Unexpected character '{c}' at line {self.line}")

    def _string(self) -> None:
        value_chars = []
        while self._peek() != '"' and not self._is_at_end():
            ch = self._advance()
            if ch == "\n":
                self.line += 1
            value_chars.append(ch)

        if self._is_at_end():
            raise LexError(f"LexError: Unterminated string at line {self.line}")

        # closing quote
        self._advance()

        literal = "".join(value_chars)
        self._add_token(TokenType.STRING, literal)

    def _number(self, first_char: str) -> None:
        digits = [first_char]
        while self._peek().isdigit():
            digits.append(self._advance())

        text = "".join(digits)
        try:
            value = int(text)
        except ValueError as exc:
            raise LexError(f"LexError: Invalid integer literal '{text}' at line {self.line}") from exc

        self._add_token(TokenType.INT, value)

    def _identifier_or_keyword(self, first_char: str) -> None:
        ident_chars = [first_char]
        while True:
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                ident_chars.append(self._advance())
            else:
                break

        text = "".join(ident_chars)

        # 使用字典缓存关键词
        if text in self.keywords:
            value = self.keywords[text]
            if isinstance(value, tuple):
                self._add_token(value[0], value[1])
            else:
                self._add_token(value)
        # Identifier
        else:
            self._add_token(TokenType.IDENTIFIER, text)

