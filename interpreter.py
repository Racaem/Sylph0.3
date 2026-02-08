from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .ast import (
    Stmt, OutputStmt, Expr, Literal, Identifier, BinaryExpr, CallExpr,
    DefStmt, LetStmt, IfStmt, ReturnStmt, UseStmt
)
from .dll import DllManager, convert_sylph_to_c, convert_c_to_sylph, get_c_type
from .errors import RuntimeError
from .lexer import Lexer
from .parser import Parser


class Environment:
    def __init__(self, enclosing: Optional[Environment] = None) -> None:
        self.values: Dict[str, Any] = {}  # 全局变量和非局部变量
        self.locals: Dict[str, int] = {}  # 局部变量到偏移量的映射
        self.local_values: List[Any] = []  # 局部变量值，通过偏移量访问
        self.enclosing = enclosing

    def get(self, name: str) -> Any:
        # 首先检查是否为局部变量
        if name in self.locals:
            return self.local_values[self.locals[name]]
        # 然后检查是否为当前环境的变量
        if name in self.values:
            return self.values[name]
        # 最后检查父环境
        if self.enclosing is not None:
            return self.enclosing.get(name)
        raise RuntimeError(f"RuntimeError: Undefined variable '{name}'")

    def set(self, name: str, value: Any) -> None:
        # 首先检查是否为局部变量
        if name in self.locals:
            self.local_values[self.locals[name]] = value
            return
        # 然后检查是否为当前环境的变量
        if name in self.values:
            self.values[name] = value
            return
        # 最后检查父环境
        if self.enclosing is not None:
            self.enclosing.set(name, value)
            return
        # 如果都不是，在当前环境定义
        self.values[name] = value

    def define(self, name: str, value: Any) -> None:
        # 定义变量时，默认添加到values中
        self.values[name] = value

    def define_local(self, name: str, value: Any) -> None:
        # 定义局部变量，添加到局部变量表中
        if name not in self.locals:
            self.locals[name] = len(self.local_values)
            self.local_values.append(value)
        else:
            self.local_values[self.locals[name]] = value


class Bytecode:
    # Bytecode instructions
    PUSH = 1
    ADD = 2
    SUB = 3
    MUL = 4
    LT = 5
    LOAD_VAR = 6
    CALL_FUNC = 7


class BytecodeCompiler:
    def __init__(self) -> None:
        self.bytecode: List[int] = []
        self.constants: List[Any] = []

    def compile(self, expr: Expr) -> tuple[List[int], List[Any]]:
        self.bytecode.clear()
        self.constants.clear()
        self._compile_expr(expr)
        return self.bytecode, self.constants

    def _compile_expr(self, expr: Expr) -> None:
        if expr.type_tag == "Literal":
            # Push literal value
            idx = self._add_constant(expr.value)
            self.bytecode.append(Bytecode.PUSH)
            self.bytecode.append(idx)
        elif expr.type_tag == "Identifier":
            # Load variable
            self.bytecode.append(Bytecode.LOAD_VAR)
            self.bytecode.append(self._add_constant(expr.name))
        elif expr.type_tag == "BinaryExpr":
            # 检查是否为常量表达式
            if expr.left.type_tag == "Literal" and expr.right.type_tag == "Literal":
                # 预计算常量表达式
                left_value = expr.left.value
                right_value = expr.right.value
                if expr.operator == '+':
                    result = left_value + right_value
                elif expr.operator == '-':
                    result = left_value - right_value
                elif expr.operator == '*':
                    result = left_value * right_value
                elif expr.operator == '<':
                    result = left_value < right_value
                else:
                    raise RuntimeError(f"RuntimeError: Unsupported operator '{expr.operator}'")
                # 生成PUSH指令，推送计算结果
                idx = self._add_constant(result)
                self.bytecode.append(Bytecode.PUSH)
                self.bytecode.append(idx)
            else:
                # 不是常量表达式，编译为正常的计算指令
                # Compile left and right operands first
                self._compile_expr(expr.left)
                self._compile_expr(expr.right)
                # Then compile the operator
                if expr.operator == '+':
                    self.bytecode.append(Bytecode.ADD)
                elif expr.operator == '-':
                    self.bytecode.append(Bytecode.SUB)
                elif expr.operator == '*':
                    self.bytecode.append(Bytecode.MUL)
                elif expr.operator == '<':
                    self.bytecode.append(Bytecode.LT)
                else:
                    raise RuntimeError(f"RuntimeError: Unsupported operator '{expr.operator}'")
        elif expr.type_tag == "CallExpr":
            # Compile the argument if it exists
            if expr.argument is not None:
                self._compile_expr(expr.argument)
            # Then call the function
            self.bytecode.append(Bytecode.CALL_FUNC)
            self.bytecode.append(self._add_constant(expr.callee))
        else:
            raise RuntimeError(f"RuntimeError: Unexpected expression type {expr.type_tag}")

    def _add_constant(self, value: Any) -> int:
        if value not in self.constants:
            self.constants.append(value)
        return self.constants.index(value)


class BytecodeInterpreter:
    def __init__(self, interpreter: 'Interpreter') -> None:
        self.interpreter = interpreter

    def interpret(self, bytecode: List[int], constants: List[Any], env: Environment) -> Any:
        stack: List[Any] = []
        ip = 0  # Instruction pointer

        while ip < len(bytecode):
            instr = bytecode[ip]
            ip += 1

            if instr == Bytecode.PUSH:
                # Push constant onto stack
                const_idx = bytecode[ip]
                ip += 1
                stack.append(constants[const_idx])
            elif instr == Bytecode.ADD:
                # Add top two values
                if len(stack) < 2:
                    raise RuntimeError("RuntimeError: Stack underflow in ADD operation")
                right = stack.pop()
                left = stack.pop()
                stack.append(left + right)
            elif instr == Bytecode.SUB:
                # Subtract top two values
                if len(stack) < 2:
                    raise RuntimeError("RuntimeError: Stack underflow in SUB operation")
                right = stack.pop()
                left = stack.pop()
                stack.append(left - right)
            elif instr == Bytecode.MUL:
                # Multiply top two values
                if len(stack) < 2:
                    raise RuntimeError("RuntimeError: Stack underflow in MUL operation")
                right = stack.pop()
                left = stack.pop()
                stack.append(left * right)
            elif instr == Bytecode.LT:
                # Compare top two values
                if len(stack) < 2:
                    raise RuntimeError("RuntimeError: Stack underflow in LT operation")
                right = stack.pop()
                left = stack.pop()
                stack.append(left < right)
            elif instr == Bytecode.LOAD_VAR:
                # Load variable
                var_name = constants[bytecode[ip]]
                ip += 1
                stack.append(env.get(var_name))
            elif instr == Bytecode.CALL_FUNC:
                # Call function
                func_name = constants[bytecode[ip]]
                ip += 1
                # Check if the function is parameterless by looking it up
                func_info = env.get(func_name)
                if isinstance(func_info, dict) and func_info.get('param') is None:
                    # Parameterless function
                    arg = None
                else:
                    # Function with parameter
                    if len(stack) < 1:
                        raise RuntimeError("RuntimeError: Stack underflow in CALL_FUNC operation")
                    arg = stack.pop()
                # Delegate to the main interpreter's function call handling
                # Create a dummy CallExpr for the function call
                class DummyCallExpr:
                    def __init__(self, callee, argument):
                        self.callee = callee
                        self.argument = argument
                result = self.interpreter._evaluate_func_call(DummyCallExpr(func_name, Literal(arg) if arg is not None else None), env)
                stack.append(result)
            else:
                raise RuntimeError(f"RuntimeError: Unknown bytecode instruction {instr}")

        if not stack:
            raise RuntimeError("RuntimeError: Empty stack after bytecode execution")
        return stack[-1]


class Interpreter:
    def __init__(self) -> None:
        self.globals = Environment()
        self.bytecode_compiler = BytecodeCompiler()
        self.bytecode_interpreter = BytecodeInterpreter(self)
        # 环境对象池，用于复用Environment对象
        self.env_pool = []
        # DLL管理器
        self.dll_manager = DllManager()
        # 注册DLL相关函数到全局环境
        self._register_dll_functions()

    def interpret(self, stmts: List[Stmt]) -> None:
        for stmt in stmts:
            self._execute(stmt, self.globals)

    def _execute(self, stmt: Stmt, env: Environment) -> Any:
        if stmt.type_tag == "OutputStmt":
            value = self._evaluate(stmt.expr, env)
            print(value)
        elif stmt.type_tag == "DefStmt":
            # 存储函数信息，包括参数、函数体、是否公开等信息
            self.globals.define(stmt.name, {
                'param': stmt.param,
                'body': stmt.body,
                'func': stmt,
                'is_public': stmt.is_public
            })
        elif stmt.type_tag == "LetStmt":
            value = self._evaluate(stmt.value, env)
            env.define(stmt.name, value)
        elif stmt.type_tag == "IfStmt":
            condition = self._evaluate(stmt.condition, env)
            if condition:
                for body_stmt in stmt.body:
                    self._execute(body_stmt, env)
        elif stmt.type_tag == "ReturnStmt":
            # Return statements are handled specially in function calls
            raise RuntimeError("RuntimeError: Return statement outside function")
        elif stmt.type_tag == "UseStmt":
            # 处理导入语句
            self._execute_use_stmt(stmt, env)
        else:
            raise RuntimeError(f"RuntimeError: Unexpected statement type {stmt.type_tag}")

    def _execute_use_stmt(self, stmt: UseStmt, env: Environment) -> None:
        """处理导入语句，加载外部模块并导入其公开函数"""
        # 构建模块文件路径
        if not stmt.module_name.endswith('.syl'):
            module_path = Path(f"{stmt.module_name}.syl")
        else:
            module_path = Path(stmt.module_name)
        if not module_path.exists():
            raise RuntimeError(f"RuntimeError: Module '{stmt.module_name}' not found")
        
        # 读取模块文件内容
        source = module_path.read_text(encoding="utf-8")
        
        # 解析模块文件
        lexer = Lexer(source)
        tokens = lexer.scan_tokens()
        parser = Parser(tokens, str(module_path))
        program = parser.parse()
        
        # 执行模块中的语句，但只导入公开函数，不执行其他语句
        for module_stmt in program.statements:
            if module_stmt.type_tag == "DefStmt":
                # 只导入公开函数
                if module_stmt.is_public:
                    # 检查是否有命名冲突
                    if module_stmt.name in env.values:
                        raise RuntimeError(f"RuntimeError: Name conflict: function '{module_stmt.name}' already exists")
                    # 在当前环境中定义该函数
                    env.define(module_stmt.name, {
                        'param': module_stmt.param,
                        'body': module_stmt.body,
                        'func': module_stmt,
                        'is_public': True
                    })
            # 对于let语句，我们也需要执行以确保模块内部的变量定义正确
            elif module_stmt.type_tag == "LetStmt":
                # 暂时不执行，因为我们只关心函数导入
                pass
            # 忽略其他语句，如out语句
            else:
                pass

    def _evaluate(self, expr: Expr, env: Environment) -> Any:
        # 暂时使用递归求值方式，确保基本功能正常
        if expr.type_tag == "Literal":
            return expr.value
        elif expr.type_tag == "Identifier":
            # 处理成员访问 (object.member)
            if '.' in expr.name:
                parts = expr.name.split('.')
                obj = env.get(parts[0])
                for part in parts[1:]:
                    obj = getattr(obj, part)
                return obj
            return env.get(expr.name)
        elif expr.type_tag == "BinaryExpr":
            left = self._evaluate(expr.left, env)
            right = self._evaluate(expr.right, env)
            if expr.operator == '+':
                return left + right
            elif expr.operator == '-':
                return left - right
            elif expr.operator == '*':
                return left * right
            elif expr.operator == '<':
                return left < right
            raise RuntimeError(f"RuntimeError: Unsupported operator '{expr.operator}'")
        elif expr.type_tag == "CallExpr":
            # 处理方法调用 (object.method(args))
            if '.' in expr.callee:
                parts = expr.callee.split('.')
                obj_name = parts[0]
                method_name = parts[1]
                obj = env.get(obj_name)
                method = getattr(obj, method_name)
                args = self._evaluate(expr.argument, env)
                
                # Extract values from Literal objects if needed
                processed_args = []
                if isinstance(args, list):
                    for arg in args:
                        if hasattr(arg, 'value'):
                            processed_args.append(arg.value)
                        else:
                            processed_args.append(arg)
                else:
                    if hasattr(args, 'value'):
                        processed_args = [args.value]
                    else:
                        processed_args = [args]
                
                # Call the method with processed arguments
                return method(*processed_args)
            # 尝试将CallExpr作为函数调用处理
            try:
                return self._evaluate_func_call(expr, env)
            except RuntimeError as e:
                # 如果函数不存在，尝试将其作为变量访问
                if f"'{expr.callee}' is not a function" in str(e):
                    return env.get(expr.callee)
                raise e
        raise RuntimeError(f"RuntimeError: Unexpected expression type {expr.type_tag}")

    def _register_dll_functions(self) -> None:
        """Register DLL-related functions to the global environment."""
        # 注册load函数 (new syntax)
        self.globals.define('load', {
            'param': 'dll_path',
            'body': [],
            'func': None,
            'is_public': False,
            'native': lambda dll_path: self._load_dll(dll_path)
        })
    
    def _load_dll(self, dll_path: str) -> Any:
        """
        Load a DLL and return a wrapper object that supports dot notation for function calls.
        
        Args:
            dll_path: Path to the DLL file
            
        Returns:
            Any: A wrapper object with methods corresponding to DLL functions
        """
        import ctypes
        import os
        
        # Get absolute path if relative path is provided
        if not os.path.isabs(dll_path):
            dll_path = os.path.abspath(dll_path)
        
        # Load the DLL
        dll = ctypes.CDLL(dll_path)
        
        # Create a wrapper class that supports dot notation
        class DllWrapper:
            def __init__(self, dll):
                self.dll = dll
            
            def __getattr__(self, name):
                """Get a function from the DLL when accessed via dot notation."""
                func = getattr(self.dll, name)
                # Set default return type to int
                func.restype = ctypes.c_int
                # Return a wrapper that handles arguments with proper type conversion
                def wrapper(*args):
                    # Convert Python arguments to C types
                    c_args = []
                    for arg in args:
                        if isinstance(arg, int):
                            c_args.append(ctypes.c_int(arg))
                        elif isinstance(arg, str):
                            c_args.append(ctypes.c_char_p(arg.encode('utf-8')))
                        elif isinstance(arg, float):
                            c_args.append(ctypes.c_float(arg))
                        elif isinstance(arg, bool):
                            c_args.append(ctypes.c_bool(arg))
                        else:
                            c_args.append(arg)
                    # Call the function with converted arguments
                    result = func(*c_args)
                    # Convert result back to Python type if needed
                    return result
                return wrapper
        
        return DllWrapper(dll)

    def _evaluate_func_call(self, expr: CallExpr, env: Environment) -> Any:
        # 从全局环境中查找函数
        func_info = self.globals.get(expr.callee)
        if not isinstance(func_info, dict) or 'param' not in func_info or 'body' not in func_info:
            raise RuntimeError(f"RuntimeError: '{expr.callee}' is not a function")
        
        # 检查是否为本地函数
        if 'native' in func_info:
            # 处理本地函数调用
            if func_info['param'] is None:
                # 无参函数
                return func_info['native']()
            else:
                # 有参函数
                arg_value = self._evaluate(expr.argument, env)
                return func_info['native'](arg_value)
        
        # 检查是否有缓存
        if 'cache' not in func_info:
            func_info['cache'] = {}
        
        # 处理无参函数
        if func_info['param'] is None:
            # 检查缓存（无参函数使用None作为缓存键）
            if None in func_info['cache']:
                return func_info['cache'][None]
            
            # 从环境对象池中获取Environment对象，或创建新的
            if self.env_pool:
                local_env = self.env_pool.pop()
                local_env.enclosing = env  # 使用传入的env作为父环境，而不是self.globals
                local_env.values.clear()
                local_env.locals.clear()
                local_env.local_values.clear()
            else:
                local_env = Environment(env)  # 使用传入的env作为父环境，而不是self.globals
            
            # Execute the function body
            result = None
            for stmt in func_info['body']:
                if stmt.type_tag == "ReturnStmt":
                    result = self._evaluate(stmt.expr, local_env)
                    break
                elif stmt.type_tag == "LetStmt":
                    value = self._evaluate(stmt.value, local_env)
                    # 将函数体内定义的变量也定义为局部变量，使用局部变量表
                    local_env.define_local(stmt.name, value)
                elif stmt.type_tag == "IfStmt":
                    condition = self._evaluate(stmt.condition, local_env)
                    if condition:
                        for body_stmt in stmt.body:
                            if body_stmt.type_tag == "ReturnStmt":
                                result = self._evaluate(body_stmt.expr, local_env)
                                # 缓存结果
                                func_info['cache'][None] = result
                                # 将Environment对象放回对象池
                                self.env_pool.append(local_env)
                                return result
                            elif body_stmt.type_tag == "LetStmt":
                                value = self._evaluate(body_stmt.value, local_env)
                                # 将if语句体内定义的变量也定义为局部变量，使用局部变量表
                                local_env.define_local(body_stmt.name, value)
                            else:
                                self._execute(body_stmt, local_env)
                else:
                    self._execute(stmt, local_env)
            
            # 缓存结果
            func_info['cache'][None] = result
            
            # 将Environment对象放回对象池
            self.env_pool.append(local_env)
            
            return result
        else:
            # 处理有参函数
            # 首先创建局部环境，然后定义参数，最后评估参数
            # 这样在评估递归调用的参数时，参数已经被定义
            
            # 从环境对象池中获取Environment对象，或创建新的
            if self.env_pool:
                local_env = self.env_pool.pop()
                local_env.enclosing = env  # 使用传入的env作为父环境，而不是self.globals
                local_env.values.clear()
                local_env.locals.clear()
                local_env.local_values.clear()
            else:
                local_env = Environment(env)  # 使用传入的env作为父环境，而不是self.globals
            
            # 评估参数，使用当前环境作为环境
            # 这样在评估参数时，不会尝试访问尚未定义的局部变量
            # 注意：对于递归调用，expr.argument可能会引用当前函数的参数
            # 但是，在这个阶段，参数还没有被定义为局部变量，所以会在父环境中查找
            # 这是正确的行为，因为递归调用的参数是基于当前函数的参数计算的
            arg_value = self._evaluate(expr.argument, env)
            
            # 将参数定义为局部变量，使用评估后的值
            # 这样在执行函数体时，参数可以在局部环境中找到
            local_env.define_local(func_info['param'], arg_value)
            
            # 检查缓存
            if arg_value in func_info['cache']:
                return func_info['cache'][arg_value]
            
            # 确保参数已正确定义
            if func_info['param'] not in local_env.locals:
                raise RuntimeError(f"RuntimeError: Parameter '{func_info['param']}' not defined")
            
            # Execute the function body
            result = None
            for stmt in func_info['body']:
                if stmt.type_tag == "ReturnStmt":
                    result = self._evaluate(stmt.expr, local_env)
                    break
                elif stmt.type_tag == "LetStmt":
                    # 对于函数体内的函数调用，使用local_env作为环境
                    # 这样递归调用时，参数可以在局部环境中找到
                    value = self._evaluate(stmt.value, local_env)
                    # 将函数体内定义的变量也定义为局部变量，使用局部变量表
                    local_env.define_local(stmt.name, value)
                elif stmt.type_tag == "IfStmt":
                    condition = self._evaluate(stmt.condition, local_env)
                    if condition:
                        for body_stmt in stmt.body:
                            if body_stmt.type_tag == "ReturnStmt":
                                result = self._evaluate(body_stmt.expr, local_env)
                                # 缓存结果
                                func_info['cache'][arg_value] = result
                                # 将Environment对象放回对象池
                                self.env_pool.append(local_env)
                                return result
                            elif body_stmt.type_tag == "LetStmt":
                                value = self._evaluate(body_stmt.value, local_env)
                                # 将if语句体内定义的变量也定义为局部变量，使用局部变量表
                                local_env.define_local(body_stmt.name, value)
                            else:
                                self._execute(body_stmt, local_env)
                else:
                    self._execute(stmt, local_env)
            
            # 缓存结果
            func_info['cache'][arg_value] = result
            
            # 将Environment对象放回对象池
            self.env_pool.append(local_env)
            
            return result
