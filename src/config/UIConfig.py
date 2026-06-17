from pydantic.main import BaseModel


class UIConfig(BaseModel):
    streaming: bool = True
