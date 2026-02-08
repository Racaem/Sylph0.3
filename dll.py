from __future__ import annotations

import ctypes
import os
from typing import Any, Dict, Optional, List

from .errors import RuntimeError


class DllManager:
    """Manages DLL loading, unloading, and function calls."""
    
    def __init__(self):
        self.loaded_dlls: Dict[str, ctypes.CDLL] = {}
    
    def load_dll(self, dll_path: str) -> str:
        """
        Load a DLL and return a handle identifier.
        
        Args:
            dll_path: Path to the DLL file
            
        Returns:
            str: A unique identifier for the loaded DLL
            
        Raises:
            RuntimeError: If the DLL cannot be loaded
        """
        try:
            # Check if DLL exists
            if not os.path.exists(dll_path):
                raise RuntimeError(f"RuntimeError: DLL file not found: {dll_path}")
            
            # Load the DLL
            dll = ctypes.CDLL(dll_path)
            
            # Generate a unique identifier
            dll_id = f"dll_{len(self.loaded_dlls)}"
            
            # Store the DLL
            self.loaded_dlls[dll_id] = dll
            
            return dll_id
            
        except Exception as e:
            raise RuntimeError(f"RuntimeError: Failed to load DLL: {str(e)}")
    
    def unload_dll(self, dll_id: str) -> bool:
        """
        Unload a previously loaded DLL.
        
        Args:
            dll_id: The identifier of the DLL to unload
            
        Returns:
            bool: True if the DLL was successfully unloaded
            
        Raises:
            RuntimeError: If the DLL identifier is not found
        """
        if dll_id not in self.loaded_dlls:
            raise RuntimeError(f"RuntimeError: DLL not found: {dll_id}")
        
        # Remove the DLL from the loaded list
        del self.loaded_dlls[dll_id]
        
        return True
    
    def get_function(self, dll_id: str, func_name: str, arg_types: List[type], return_type: type) -> Any:
        """
        Get a function from a loaded DLL with specified argument and return types.
        
        Args:
            dll_id: The identifier of the DLL
            func_name: The name of the function to get
            arg_types: List of argument types
            return_type: Return type
            
        Returns:
            Any: A callable function object
            
        Raises:
            RuntimeError: If the DLL or function is not found
        """
        if dll_id not in self.loaded_dlls:
            raise RuntimeError(f"RuntimeError: DLL not found: {dll_id}")
        
        dll = self.loaded_dlls[dll_id]
        
        try:
            # Get the function
            func = getattr(dll, func_name)
            
            # Set argument types and return type
            func.argtypes = arg_types
            func.restype = return_type
            
            return func
            
        except AttributeError:
            raise RuntimeError(f"RuntimeError: Function not found in DLL: {func_name}")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: Failed to get function: {str(e)}")


def convert_sylph_to_c(value: Any) -> Any:
    """
    Convert a Sylph value to a C-compatible value.
    
    Args:
        value: The Sylph value to convert
        
    Returns:
        Any: The C-compatible value
    """
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return value
    elif isinstance(value, str):
        return ctypes.c_char_p(value.encode('utf-8'))
    elif isinstance(value, bool):
        return ctypes.c_bool(value)
    elif value is None:
        return None
    else:
        raise RuntimeError(f"RuntimeError: Unsupported type for C conversion: {type(value)}")


def convert_c_to_sylph(value: Any) -> Any:
    """
    Convert a C value to a Sylph-compatible value.
    
    Args:
        value: The C value to convert
        
    Returns:
        Any: The Sylph-compatible value
    """
    if isinstance(value, ctypes.c_char_p):
        return value.value.decode('utf-8') if value.value else ""
    elif isinstance(value, ctypes.c_bool):
        return bool(value)
    else:
        return value


def get_c_type(type_name: str) -> type:
    """
    Get the corresponding C type for a type name.
    
    Args:
        type_name: The name of the type
        
    Returns:
        type: The corresponding C type
        
    Raises:
        RuntimeError: If the type is not supported
    """
    type_map = {
        "int": ctypes.c_int,
        "float": ctypes.c_float,
        "double": ctypes.c_double,
        "char*": ctypes.c_char_p,
        "bool": ctypes.c_bool,
        "void": None
    }
    
    if type_name not in type_map:
        raise RuntimeError(f"RuntimeError: Unsupported C type: {type_name}")
    
    return type_map[type_name]
