import torch
from typing import Union, List, Dict, Tuple

# ANSI Color codes
class Colors:
    GREEN = "\033[32m"      # For shape_info
    PURPLE = "\033[35m"     # For dtype
    GRAY = "\033[90m"       # For device
    BLUE = "\033[34m"        # For requires_grad
    RESET = "\033[0m"       # Reset color

class TensorInfo:
    def __init__(self, x: torch.Tensor, detailed_info=False):
        if not isinstance(x, torch.Tensor):
            raise ValueError("x must be a torch.Tensor")
        self.x = x
        self.detailed_info = detailed_info
        self.shape_info = f"{Colors.GREEN}<" + "  ".join(str(dim) for dim in self.x.shape) + f">{Colors.RESET}"
    
    def __repr__(self):
        if not self.detailed_info:
            return f"{self.shape_info}"
        else:
            return (
                f"<{self.shape_info}, "
                f"{Colors.PURPLE}{self.x.dtype}{Colors.RESET}, "
                f"{Colors.GRAY}{self.x.device}{Colors.RESET}, "
                f"{Colors.BLUE}{self.x.requires_grad}{Colors.RESET}>"
            )

class ObjectSkeleton:
    MAX_RECURSION_DEPTH = 1000  # Define a reasonable limit
    
    def __init__(self, x: Union[List, Tuple, Dict, torch.Tensor], detailed_info=False, current_depth=0):
        if current_depth > self.MAX_RECURSION_DEPTH:
            raise RecursionError("Maximum recursion depth exceeded in ObjectSkeleton")
        
        self.current_depth = current_depth  # Track recursion depth
        
        if isinstance(x, torch.Tensor):
            self.tensor_like = TensorInfo(x, detailed_info=detailed_info)
        elif isinstance(x, (list, tuple)):
            # Determine if the input is a list or tuple to preserve the type
            container_type = list if isinstance(x, list) else tuple
            # Recursively wrap each element
            wrapped_elements = [ObjectSkeleton(i, detailed_info=detailed_info, current_depth=current_depth + 1) for i in x]
            self.tensor_like = container_type(wrapped_elements)
        elif isinstance(x, dict):
            self.tensor_like = {k: ObjectSkeleton(v, detailed_info=detailed_info, current_depth=current_depth + 1) for k, v in x.items()}
        else:
            self.tensor_like = x
    
    def _is_primitive(self, obj):
        # Corrected isinstance usage
        return not isinstance(obj, (list, tuple, dict))
    
    def _format_primitive(self, obj):
        if isinstance(obj, str):
            return f'"{obj}"'
        return str(obj)
    
    def _format_with_indent(self, obj, level=0, indent=4):
        import sys
        if level > self.MAX_RECURSION_DEPTH:
            raise RecursionError("Maximum recursion depth exceeded in _format_with_indent")
        
        spaces = " " * (level * indent)
        next_level_spaces = " " * ((level + 1) * indent)
        
        if isinstance(obj, ObjectSkeleton):
            return self._format_with_indent(obj.tensor_like, level, indent)
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return "[]" if isinstance(obj, list) else "()"
            # Check if all child objects are primitive
            _all_child_objects_are_primitive_types = all([
                self._is_primitive(x.tensor_like if isinstance(x, ObjectSkeleton) else x) for x in obj
            ])
            # Choose brackets based on the type
            open_bracket, close_bracket = ("[", "]") if isinstance(obj, list) else ("(", ")")
            if _all_child_objects_are_primitive_types:
                elements = ', '.join([
                    self._format_primitive(x.tensor_like if isinstance(x, ObjectSkeleton) else x) for x in obj
                ])
                return f"{open_bracket}{elements}{close_bracket}"
            else:
                result = f"{open_bracket}\n"
                for i, item in enumerate(obj):
                    result += next_level_spaces + self._format_with_indent(item, level + 1, indent)
                    if i < len(obj) - 1:
                        result += ","
                    result += "\n"
                result += spaces + close_bracket
                return result
        elif isinstance(obj, dict):
            if not obj:
                return "{}"
            result = "{\n"
            items = list(obj.items())
            for i, (k, v) in enumerate(items):
                formatted_value = self._format_with_indent(v, level + 1, indent)
                formatted_key = self._format_primitive(k)
                result += f"{next_level_spaces}{formatted_key}: {formatted_value}"
                if i < len(items) - 1:
                    result += ","
                result += "\n"
            result += spaces + "}"
            return result
        elif isinstance(obj, TensorInfo):
            return str(obj)
        else:
            return self._format_primitive(obj)
                
    
    def __repr__(self):
        return self._format_with_indent(self.tensor_like)