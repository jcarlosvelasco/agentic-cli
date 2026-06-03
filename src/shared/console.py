from contextlib import asynccontextmanager
from typing import AsyncGenerator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


def display_welcome() -> None:
    console.print()
    console.rule("[bold blue]coding-agent[/]")
    console.print()


def display_user_message(content: str) -> None:
    panel = Panel(
        content,
        title="[blue]You[/]",
        title_align="left",
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)


def display_assistant_message(content: str) -> None:
    panel = Panel(
        Markdown(content),
        title="[green]Agent[/]",
        title_align="left",
        border_style="green",
        padding=(0, 1),
    )
    console.print(panel)


def display_tool_call(name: str, args: object) -> None:
    console.print(f"[yellow]🔧 {name}[/] [dim]({args})[/dim]")


def display_tool_result(data: dict | str | None) -> None:
    if data:
        console.print(f"[dim]{data}[/dim]")


def display_compacting(count: int) -> None:
    console.print(f"[dim]Compacting {count} messages...[/dim]")


def get_user_input() -> str:
    return Prompt.ask("[bold blue]You[/]")


def confirm_execution(tool_name: str, tool_input: object) -> bool:
    return Confirm.ask(
        f"[yellow]Execute {tool_name}[/] with [dim]{tool_input}[/dim]?",
        default=False,
    )


@asynccontextmanager
async def thinking_spinner() -> AsyncGenerator[None, None]:
    with console.status("[yellow]Thinking...[/]", spinner="dots"):
        yield
