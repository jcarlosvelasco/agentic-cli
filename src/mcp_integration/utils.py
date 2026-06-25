from typing import Any, Dict, List, Type, cast

from pydantic import BaseModel, create_model


def _json_type_to_python(json_type: str) -> type[Any]:
    mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, cast(type[Any], Any))


def build_args_schema(input_schema: Dict[str, Any]) -> Type[BaseModel]:
    properties: Dict[str, Any] = input_schema.get("properties", {})
    required: List[str] = input_schema.get("required", [])

    field_definitions: Dict[str, Any] = {}
    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        python_type = _json_type_to_python(json_type)

        if field_name in required:
            field_definitions[field_name] = (python_type, ...)
        else:
            field_definitions[field_name] = (python_type | None, None)

    return create_model("MCPToolArgs", **field_definitions)
