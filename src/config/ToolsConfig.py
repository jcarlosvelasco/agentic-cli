from pydantic.main import BaseModel


class ToolsConfig(BaseModel):
    enabled: bool = True
