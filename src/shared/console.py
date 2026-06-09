from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner

console = Console()

_session = PromptSession(history=FileHistory(str(Path.home() / ".opencode_history")))


def display_welcome() -> None:
    console.print()
    console.rule("[bold blue]coding-agent[/]")
    console.print()


def display_warning(message: str) -> None:
    console.print(f"\n[yellow]{message}[/]")


def display_user_message(content: str) -> None:
    panel = Panel(
        content,
        title="[blue]You[/]",
        title_align="left",
        border_style="blue",
        padding=(0, 1),
    )
    console.print(panel)


def display_assistant_message(agent_name: str, content: str) -> None:
    panel = Panel(
        Markdown(content),
        title=f"[green]{agent_name}[/]",
        title_align="left",
        border_style="green",
        padding=(0, 1),
    )
    console.print(panel)


def display_tool_call(agent_name: str, name: str, args: object) -> None:
    console.print(f"\n[green]{agent_name}[/] [yellow]🔧 {name}[/] [dim]({args})[/dim]")


def display_tool_result(agent_name: str, data: dict | str | None) -> None:
    if data:
        console.print(f"\n[green]{agent_name}[/] [dim]{data}[/dim]")


def display_compacting(count: int) -> None:
    console.print(f"\n[dim]Compacting {count} messages...[/dim]")


async def get_user_input() -> str:
    return await _session.prompt_async("You: ") or ""


def confirm_execution(tool_name: str, tool_input: object) -> bool:
    return Confirm.ask(
        f"\n[yellow]Execute {tool_name}[/] with [dim]{tool_input}[/dim]?",
        default=False,
    )


@asynccontextmanager
async def streaming_panel(
    agent_name: str, initial: str = ""
) -> AsyncGenerator[Callable[[str], None], None]:
    spinner = Spinner("dots", text="[yellow]Thinking...[/]")
    with Live(spinner, refresh_per_second=10, vertical_overflow="visible") as live:

        def update(content: str) -> None:
            panel = Panel(
                Markdown(content),
                title=f"[green]{agent_name}[/]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
            live.update(panel)

        yield update


@asynccontextmanager
async def thinking_spinner() -> AsyncGenerator[None, None]:
    with console.status("[yellow]Thinking...[/]", spinner="dots"):
        yield
