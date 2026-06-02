from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from pydantic import BaseModel

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class Tool(ABC, Generic[ArgsT]):
    name: str
    description: str
    args_schema: Type[ArgsT] | None

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[ArgsT] | None = None,
    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema

    @abstractmethod
    def execute(self, args: ArgsT | None) -> None:
        pass
