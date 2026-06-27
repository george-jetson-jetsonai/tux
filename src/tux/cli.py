"""Command-line entry point for the ``tux`` console command."""

import argparse
import os
import subprocess
import sys
from collections.abc import Callable

from tux import __version__
from tux.client import DEFAULT_VARIANT, DEFAULTS, ModelClient, ModelClientError
from tux.modes.command import CommandSuggestion, Plan, assistant_turn, output_message
from tux.chooser import Chooser, select
from tux.config import (
    ALLOWED_KEYS,
    ConfigError,
    config_path,
    load_config,
    resolved_settings,
    set_value,
)
from tux.provision import ProvisionResult, provision
from tux.runner import CommandRunner, append_run, run_command
from tux.safety import destructive_reason
from tux.state import clear_thread, load_thread, save_thread

DESCRIPTION = (
    "A terminal-native Linux assistant that proposes commands in plain "
    "English; you decide whether to run them."
)

EXAMPLE = """\
example:
  Ask tux in plain English what you want to do, and it suggests the command:

    you:  how do I see which processes are using the most memory?
    tux:  ps aux --sort=-%mem | head -10
          (lists the top ten processes ranked by memory usage)

tux proposes the command; you decide whether to run it.\
"""

ASK_DESCRIPTION = "Ask tux a plain-English question about working with your system."

#: Prompt shown before each interactive turn.
SESSION_PROMPT = "you: "

#: One-line intro printed when the interactive session starts.
SESSION_INTRO = (
    "tux interactive session. Ask a question, then ask follow-ups in context.\n"
    "Press Ctrl-D or type 'exit' to quit."
)

#: Inputs that end the interactive session, matched case-insensitively.
EXIT_WORDS = frozenset({"exit", "quit"})

#: Per-step choices shown on the active step of a walked plan. Dismiss is listed
#: first so it is both the highlighted default and the fallback on any abort
#: (Ctrl-D, Escape, interrupt), keeping the safe choice the default; clarify
#: lets the user re-plan the remaining steps with a free-text question.
RUN_CHOICES = ("Dismiss", "Run", "Clarify")

#: Index in :data:`RUN_CHOICES` that abandons the rest of the plan.
DISMISS_CHOICE = 0

#: Index in :data:`RUN_CHOICES` that means run the active step's command.
RUN_CHOICE = 1

#: Index in :data:`RUN_CHOICES` that re-plans the remaining steps from free text.
CLARIFY_CHOICE = 2

#: Prompt shown when the user chooses clarify and tux reads their question.
CLARIFY_PROMPT = "clarify: "

#: Variant name that engages lite gating. Any other state — ``full``, unset, or a
#: user-supplied endpoint with no variant (8a's escape hatch) — leaves today's
#: full behavior unchanged; lite gating engages only on an explicit match here.
LITE_VARIANT = "lite"

#: Per-turn choices on a lite command proposal. Lite shows a single command with
#: no clarify/re-plan, so the menu is just the safe-run floor: dismiss (the
#: highlighted default and abort fallback) and run. ``RUN_CHOICE``/``DISMISS_CHOICE``
#: index into it identically to :data:`RUN_CHOICES`.
LITE_RUN_CHOICES = ("Dismiss", "Run")

#: Deterministic, tux-authored line appended after a lite conversational reply,
#: steering the user back toward command lookup with a concrete example request.
#: Kept here (not model-generated) so it is testable; plain text, not TTY-gated.
LITE_STEER = (
    "tux works best when you ask it for a command — for example: "
    'tux ask "how do I find the largest files in this folder?"'
)

#: ANSI styling for the interactive, framed proposal block. Emitted only when
#: interactive; the non-TTY fallback renders the same fields as plain text with
#: no escape sequences. The title, the command, and the description each get a
#: visually distinct style, and the dim label prefixes set the two labelled lines
#: apart from the bare title.
_RESET = "\x1b[0m"
_TITLE_STYLE = "\x1b[1;33m"  # bold yellow
_LABEL_STYLE = "\x1b[2m"  # dim
_COMMAND_STYLE = "\x1b[1;36m"  # bold cyan
_WARNING_STYLE = "\x1b[1;31m"  # bold red

#: Prefix on the destructive-command warning, shared by the styled and plain
#: branches so the same warning text reaches a terminal user and a log/script.
_WARNING_PREFIX = "potentially destructive"

#: Character repeated to form the horizontal rules that frame the proposal.
_RULE_CHAR = "─"

#: A clarify reader reads one line of free text (given a prompt) so the user can
#: re-plan the remaining steps. It mirrors the run-session ``input(...)`` seam and
#: is injected in tests so the clarify path runs without a real terminal.
ClarifyReader = Callable[[str], str]


def _default_reader(prompt: str) -> str:
    """Read one line of clarify text from stdin via ``input`` (the live default)."""
    return input(prompt)


def _resolve_variant() -> str:
    """Return the configured variant, falling back to the built-in default.

    Lite gating keys on this: only an explicit ``variant = "lite"`` engages it.
    An unset variant — including 8a's escape hatch where the user named their own
    ``endpoint`` with no ``variant`` — resolves to the default (full) so existing
    users and the escape hatch keep today's behavior untouched.

    Raises:
        ConfigError: If the config file exists but is not valid TOML.
    """
    return load_config().get("variant", DEFAULT_VARIANT)

CONFIG_DESCRIPTION = "Inspect and change the endpoint and model tux talks to."

#: Help line shown in ``tux --help``; points the reader at the fuller help.
CONFIG_HELP = f"{CONFIG_DESCRIPTION} Run `config --help` for more."

PROVISION_DESCRIPTION = (
    "Assess this machine's hardware, ensure the Ollama runtime, pull a model "
    "sized to the host, and point tux's config at the local endpoint."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``tux`` command."""
    parser = argparse.ArgumentParser(
        prog="tux",
        description=DESCRIPTION,
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tux {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    ask_parser = subparsers.add_parser(
        "ask",
        help=ASK_DESCRIPTION,
        description=ASK_DESCRIPTION,
    )
    ask_parser.add_argument(
        "question",
        nargs="?",
        help="The plain-English question to ask tux. Omit it to start an "
        "interactive session where you can ask follow-up questions.",
    )
    ask_parser.add_argument(
        "--new",
        action="store_true",
        help="Start a fresh conversation in this terminal, discarding any prior "
        "context from earlier questions in this shell.",
    )
    _add_config_parser(subparsers)
    _add_provision_parser(subparsers)
    return parser


def _config_description() -> str:
    """Return the ``config`` subcommand description, noting where the file lives."""
    return f"{CONFIG_DESCRIPTION}\n\nThe config file lives at {config_path()}."


def _add_config_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``config`` subcommand with its ``show`` / ``set`` / ``path`` actions."""
    config_parser = subparsers.add_parser(
        "config",
        help=CONFIG_HELP,
        description=_config_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = config_parser.add_subparsers(
        dest="config_command", metavar="action", required=True
    )
    actions.add_parser(
        "show",
        help="Show the effective endpoint and model and where each comes from.",
        description="Show the effective endpoint and model and where each comes from.",
    )
    actions.add_parser(
        "path",
        help="Print the path tux uses for its config file.",
        description="Print the path tux uses for its config file.",
    )
    set_parser = actions.add_parser(
        "set",
        help="Set a config value, creating the config file if needed.",
        description="Set a config value, creating the config file if needed.",
    )
    set_parser.add_argument("key", choices=ALLOWED_KEYS, help="The config key to set.")
    set_parser.add_argument("value", help="The value to store for the key.")


def _add_provision_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the ``provision`` subcommand for the guided, re-runnable install."""
    provision_parser = subparsers.add_parser(
        "provision",
        help=PROVISION_DESCRIPTION,
        description=PROVISION_DESCRIPTION,
    )
    provision_parser.add_argument(
        "--yes",
        action="store_true",
        help="Treat the model-download consent as already granted; do not prompt. "
        "Use for an unattended install with consent preseeded.",
    )
    provision_parser.add_argument(
        "--variant",
        choices=("lite", "full"),
        help="Pin the variant tier instead of probing hardware. The variant "
        "packages use this (e.g. tux-lite passes 'lite') so a packaged install "
        "stays on its tier regardless of the host's hardware.",
    )


def run_ask(
    question: str,
    client: ModelClient | None = None,
    *,
    new: bool = False,
    runner: CommandRunner = run_command,
    chooser: Chooser = select,
    reader: ClarifyReader = _default_reader,
) -> int:
    """Answer one ``tux ask`` turn, carrying context across invocations.

    The conversation thread for this terminal (keyed to the parent shell's PID)
    is loaded from disk, the new turn is routed to either a guided plan or a
    conversational reply, the answer is shown, and the thread is saved so the
    next ``tux ask`` from the same shell sees this turn. A command turn comes
    back as an ordered plan and, in an interactive terminal, is walked one step
    at a time through the run/dismiss/clarify surface; a non-interactive session
    prints the whole plan only. Nothing runs until the user explicitly chooses
    run on a step.

    Args:
        question: The plain-English message for this turn.
        client: Model client to use; defaults to one built from the config file.
        new: When true, discard this shell's existing thread first so the turn
            starts a fresh conversation with no prior context.
        runner: Callable that executes a chosen step; injected in tests so the
            run path is exercised without spawning a real process.
        chooser: Callable presenting the per-step menu; injected in tests so the
            choice is driven without a real terminal.
        reader: Callable reading the clarify free text; injected in tests so the
            clarify re-plan runs without a real terminal.

    Returns:
        The last run step's exit status when the user ran one; otherwise ``0`` on
        a dismissed, clarify-only, conversational, or non-interactive turn; ``1``
        if the model could not be reached or parsed, or the config is malformed.
    """
    if client is None:
        try:
            client = ModelClient.from_config()
        except ConfigError as exc:
            print(f"tux: {exc}", file=sys.stderr)
            return 1
    try:
        variant = _resolve_variant()
    except ConfigError as exc:
        print(f"tux: {exc}", file=sys.stderr)
        return 1
    ppid = os.getppid()
    if new:
        clear_thread(ppid)
        history: list[dict[str, str]] = []
    else:
        history = load_thread(ppid)
    try:
        status, assistant = _answer_turn(
            client, question, history, runner, chooser, reader, variant
        )
    except ModelClientError as exc:
        # Report and leave the stored thread untouched, so a failed turn never
        # corrupts or truncates the conversation that was already saved.
        print(f"tux: {exc}", file=sys.stderr)
        return 1
    save_thread(ppid, [*history, {"role": "user", "content": question}, assistant])
    return status


def _answer_turn(
    client: ModelClient,
    question: str,
    history: list[dict[str, str]],
    runner: CommandRunner,
    chooser: Chooser,
    reader: ClarifyReader,
    variant: str,
) -> tuple[int, dict[str, str]]:
    """Route the turn, present the answer, and return ``(status, message)``.

    The model decides the turn type: a command request goes through the
    structured ``suggest`` path, while a conversational one goes through the
    free-form ``converse`` path and is shown as prose, never walked. ``status``
    is the last run step's exit status when one was run, and ``0`` otherwise;
    ``message`` is the assistant turn (the plan or prose) to store so a follow-up
    sees this turn as context.

    In the **lite** variant a command turn is reduced to a single-command
    proposal (no multi-step overview, per-step walk, or clarify/re-plan loop) and
    a conversational reply still answers in prose but ends with a steer back
    toward command lookup. Any other variant — full, unset, or the escape hatch —
    keeps today's full behavior, conversational replies unsteered.
    """
    lite = variant == LITE_VARIANT
    if client.classify(question, history) == "command":
        plan = client.suggest(question, history)
        status, final_plan = _present_command(
            client, question, history, plan, runner, chooser, reader, lite=lite
        )
        return status, assistant_turn(final_plan)
    answer = client.converse(question, history)
    print(answer)
    if lite:
        # cli-layer append: the model's prose is saved as-is; the deterministic
        # steer is shown only, so it never leaks into the stored thread.
        print(f"\n{LITE_STEER}")
    return 0, {"role": "assistant", "content": answer}


def _present_command(
    client: ModelClient,
    question: str,
    history: list[dict[str, str]],
    plan: Plan,
    runner: CommandRunner,
    chooser: Chooser,
    reader: ClarifyReader,
    *,
    lite: bool = False,
) -> tuple[int, Plan]:
    """Present a plan and, in a terminal, walk it; return ``(status, final_plan)``.

    In the **lite** variant the turn is reduced to a single-command proposal via
    :func:`_present_single_command`, skipping the multi-step machinery entirely.
    Otherwise, in an interactive terminal the whole plan is shown up front as a
    compact, de-emphasised overview, then walked one step at a time through the
    run/dismiss/clarify surface. A non-interactive session renders every step as
    plain text (no styling, no menu) and returns ``0`` with the plan unchanged,
    never running anything, so the propose-only guarantee holds end to end. The
    returned plan is the plan as last known (after any in-walk re-plans), stored
    so the next turn sees this turn as context.
    """
    if lite:
        return _present_single_command(plan, runner, chooser)
    if not _interactive():
        _print_plan_plain(plan)
        return 0, plan
    return _walk_plan(client, question, history, plan, runner, chooser, reader)


def _present_single_command(
    plan: Plan, runner: CommandRunner, chooser: Chooser
) -> tuple[int, Plan]:
    """Present a lite command turn as a single proposal; return ``(status, plan)``.

    Lite stays lookup-first: only the first proposed command is shown — there is
    no multi-step overview, no per-step walk, and no clarify/re-plan loop — while
    the safe-run floor is unchanged. In a terminal the proposal is framed and the
    user is offered run or dismiss (dismiss the highlighted default and abort
    fallback); choosing run executes the command and logs it. A piped session
    prints the proposal plainly and runs nothing. The returned plan is the single
    proposal so the next turn sees this command as context.
    """
    if not plan:
        return 0, []
    suggestion = plan[0]
    if not _interactive():
        _print_suggestion(suggestion, styled=False)
        return 0, [suggestion]
    _print_suggestion(suggestion, styled=True)
    if chooser(LITE_RUN_CHOICES) == RUN_CHOICE:
        status, _ = runner(suggestion.command)
        append_run(suggestion.command, status)
        return status, [suggestion]
    return 0, [suggestion]


def _walk_plan(
    client: ModelClient,
    question: str,
    history: list[dict[str, str]],
    plan: Plan,
    runner: CommandRunner,
    chooser: Chooser,
    reader: ClarifyReader,
) -> tuple[int, Plan]:
    """Walk an ordered plan one step at a time; return ``(status, final_plan)``.

    The overview is printed first so the user sees the path ahead, then each step
    is shown expanded as its own framed block when the walk reaches it, with a
    run/dismiss/clarify choice offered only on that active step. On run the step's
    single command is executed, the run is logged, and — when later steps remain —
    its captured output is fed back to the model so the remaining steps are
    refined (placeholders resolved) without the user copy/pasting. On clarify the
    user's free text re-plans the remaining steps and the revised overview is
    reprinted. On dismiss (also the abort fallback) the rest of the plan is
    abandoned. Nothing runs until the user chooses run on the active step.
    """
    # Working conversation trail for in-walk re-planning: the prior turns, this
    # turn's question, then the assistant's plan. Re-plans append to it so the
    # model always sees the running context (the same history-in-body trail).
    thread = [*history, {"role": "user", "content": question}, assistant_turn(plan)]
    steps = list(plan)
    status = 0
    index = 0
    _print_overview(steps)
    while index < len(steps):
        step = steps[index]
        _print_suggestion(step, styled=True)
        choice = chooser(RUN_CHOICES)
        if choice == RUN_CHOICE:
            run_status, output = runner(step.command)
            append_run(step.command, run_status)
            status = run_status
            index += 1
            if index < len(steps):
                steps = steps[:index] + _replan_from_output(
                    client, thread, step.command, output
                )
        elif choice == CLARIFY_CHOICE:
            clarification = _read_clarification(reader)
            if clarification is None:
                break
            revised = _replan_from_clarification(client, thread, clarification)
            steps = steps[:index] + revised
            _print_overview(revised)
        else:
            break
    return status, steps


def _replan_from_output(
    client: ModelClient, thread: list[dict[str, str]], command: str, output: str
) -> Plan:
    """Feed a run step's output back and return the refined remaining steps.

    The output is appended to the running thread as context and the model
    re-plans the remaining steps so a placeholder resolves from the real output.
    The captured output reaches only the model and the user, never the run log.
    """
    message = output_message(command, output)
    revised = client.suggest(message, thread)
    thread.append({"role": "user", "content": message})
    thread.append(assistant_turn(revised))
    return revised


def _replan_from_clarification(
    client: ModelClient, thread: list[dict[str, str]], clarification: str
) -> Plan:
    """Send the clarify text to the model and return the revised remaining steps."""
    revised = client.suggest(clarification, thread)
    thread.append({"role": "user", "content": clarification})
    thread.append(assistant_turn(revised))
    return revised


def _read_clarification(reader: ClarifyReader) -> str | None:
    """Read the clarify free text, or ``None`` to abort (blank or end-of-input).

    A blank line or a Ctrl-D resolves to abort so a stray keystroke never runs a
    step and the walk falls back to the safe dismiss path.
    """
    try:
        text = reader(CLARIFY_PROMPT).strip()
    except EOFError:
        # A bare Ctrl-D leaves the prompt mid-line; finish it cleanly.
        print()
        return None
    return text or None


def _print_overview(steps: Plan) -> None:
    """Print the whole plan as a compact, de-emphasised numbered title list.

    Display only — the overview shows the path ahead and runs nothing. Each step
    is one dim numbered title line, set apart from the framed active step by a
    blank line above and below.
    """
    print()
    for number, step in enumerate(steps, start=1):
        print(f"{_LABEL_STYLE}{number}. {step.title}{_RESET}")
    print()


def _print_plan_plain(plan: Plan) -> None:
    """Print every step of the plan as plain text for a non-interactive session.

    No styling, no menu, no overview frame: each step renders as the same
    labelled title/command/description block the piped fallback already uses,
    keeping the per-step single-command shape visible to scripts.
    """
    for step in plan:
        _print_suggestion(step, styled=False)


def _interactive() -> bool:
    """Return whether tux is attached to a terminal on both stdin and stdout.

    Only then is the run/dismiss menu shown; a piped or redirected session falls
    back to printing the proposal and running nothing, preserving the
    one-shot/scripting path.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_suggestion(suggestion: CommandSuggestion, *, styled: bool) -> None:
    """Render a proposal as a title / command / description block.

    When ``styled`` (an interactive terminal), the block is framed between two
    horizontal rules, carries distinct ANSI styling per line, and gets vertical
    breathing room — a blank line above and a blank line below before the menu.
    When not styled (the piped/redirected fallback), the same three fields are
    printed as plain text with no escape sequences and no frame, keeping the
    one-shot/scripting path script-friendly. Running is offered separately and
    only after the user explicitly chooses it.

    A command that tux's static inspection flags as potentially destructive
    carries a distinct warning — bold red and set apart in the styled block,
    plain text in the fallback — so the risk is visible before the run/dismiss
    choice. A non-destructive command renders exactly as it did before.
    """
    reason = destructive_reason(suggestion.command)
    if not styled:
        print(suggestion.title)
        print(f"command: {suggestion.command}")
        print(f"description: {suggestion.description}")
        if reason is not None:
            print(f"warning: {_WARNING_PREFIX} — {reason}")
        return
    width = max(
        len(suggestion.title),
        len(f"command: {suggestion.command}"),
        len(f"description: {suggestion.description}"),
    )
    rule = _RULE_CHAR * width
    print()  # padding above the block
    print(rule)
    print(f"{_TITLE_STYLE}{suggestion.title}{_RESET}")
    print(f"{_LABEL_STYLE}command:{_RESET} {_COMMAND_STYLE}{suggestion.command}{_RESET}")
    print(f"{_LABEL_STYLE}description:{_RESET} {suggestion.description}")
    print(rule)
    if reason is not None:
        # Set apart below the frame, in bold red, so it is read before the menu.
        print(f"{_WARNING_STYLE}⚠ {_WARNING_PREFIX}: {reason}{_RESET}")
    print()  # margin below the block, before the menu


def run_session(
    client: ModelClient | None = None,
    runner: CommandRunner = run_command,
    chooser: Chooser = select,
    reader: ClarifyReader = _default_reader,
) -> int:
    """Run an interactive, multi-turn conversation, holding context in memory.

    The user types a question, sees the proposed plan, and walks it one step at a
    time through the run/dismiss/clarify surface, then asks follow-ups that build
    on the earlier turns. Each request carries the prior turns so a follow-up is
    answered in context. A step may be run (tux executes it and logs the run),
    dismissed, or clarified; nothing runs until the user chooses run. The
    accumulated context lives only for this session. The session ends on EOF or
    an exit word.

    Args:
        client: Model client to use; defaults to one built from the config file.
        runner: Callable that executes a chosen step; injected in tests so the
            run path is exercised without spawning a real process.
        chooser: Callable presenting the per-step menu; injected in tests so the
            choice is driven without a real terminal.
        reader: Callable reading the clarify free text; injected in tests so the
            clarify re-plan runs without a real terminal.

    Returns:
        ``0`` when the session ends cleanly, ``1`` if the client could not be
        built from a malformed config file.
    """
    if client is None:
        try:
            client = ModelClient.from_config()
        except ConfigError as exc:
            print(f"tux: {exc}", file=sys.stderr)
            return 1
    try:
        lite = _resolve_variant() == LITE_VARIANT
    except ConfigError as exc:
        print(f"tux: {exc}", file=sys.stderr)
        return 1
    print(SESSION_INTRO)
    # Prior turns as chat messages, oldest first; sent with each request so the
    # model answers follow-ups in context. Lives only for this session.
    history: list[dict[str, str]] = []
    while True:
        try:
            line = input(SESSION_PROMPT)
        except EOFError:
            # A bare Ctrl-D leaves the prompt mid-line; finish it cleanly.
            print()
            return 0
        question = line.strip()
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            return 0
        try:
            plan = client.suggest(question, history)
            _present_command(
                client, question, history, plan, runner, chooser, reader, lite=lite
            )
        except ModelClientError as exc:
            # Report and keep the session (and its accumulated context) alive; the
            # failed turn — whether the initial plan or a mid-walk re-plan — is not
            # added, so the thread stays uncorrupted.
            print(f"tux: {exc}", file=sys.stderr)
            continue
        history.append({"role": "user", "content": question})
        history.append(assistant_turn(plan))


def main(
    argv: list[str] | None = None,
    client: ModelClient | None = None,
    runner: CommandRunner = run_command,
    chooser: Chooser = select,
    reader: ClarifyReader = _default_reader,
) -> int:
    """Run the ``tux`` command-line interface.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]`` when ``None``.
        client: Model client for the ``ask`` flow; injected in tests so the flow
            can run without a live endpoint. Defaults to one built from the
            environment.
        runner: Callable that executes a chosen step; injected in tests so the
            run path is exercised without spawning a real process.
        chooser: Callable presenting the per-step menu; injected in tests so the
            choice is driven without a real terminal.
        reader: Callable reading the clarify free text; injected in tests so the
            clarify re-plan runs without a real terminal.

    Returns:
        Process exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ask":
        # ``--version`` and ``--help`` are handled (and exit) inside parse_args.
        # No question given starts an interactive, multi-turn session; a supplied
        # question keeps the one-shot path intact for scripting/piping.
        if args.question is None:
            return run_session(client, runner, chooser, reader)
        return run_ask(
            args.question, client, new=args.new, runner=runner, chooser=chooser,
            reader=reader,
        )
    if args.command == "config":
        return run_config(args)
    if args.command == "provision":
        return run_provision(args)
    # No subcommand was given, so show the help to keep the tool self-explanatory.
    parser.print_help()
    return 0


def run_config(args: argparse.Namespace) -> int:
    """Dispatch a ``tux config`` action and report the result.

    Args:
        args: Parsed arguments carrying ``config_command`` and, for ``set``, the
            ``key`` and ``value`` to persist.

    Returns:
        ``0`` on success, ``1`` if the config file is malformed or the key is
        rejected.
    """
    if args.config_command == "show":
        return _config_show()
    if args.config_command == "set":
        return _config_set(args.key, args.value)
    # ``path`` is the only remaining action; the parser rejects anything else.
    print(config_path())
    return 0


def _config_show() -> int:
    """Print each effective setting with whether it came from the file or default."""
    try:
        settings = resolved_settings(DEFAULTS)
    except ConfigError as exc:
        print(f"tux: {exc}", file=sys.stderr)
        return 1
    for key, value, source in settings:
        print(f"{key} = {value}  ({source})")
    return 0


def _config_set(key: str, value: str) -> int:
    """Persist ``key = value`` to the config file, reporting any rejection."""
    try:
        set_value(key, value)
    except ConfigError as exc:
        print(f"tux: {exc}", file=sys.stderr)
        return 1
    return 0


def run_provision(args: argparse.Namespace) -> int:
    """Run the guided, re-runnable provisioning and report what it did.

    Consent is interactive only when tux is attached to a terminal; a piped or
    redirected (unattended) run never prompts — it defers the model pull to first
    run unless ``--yes`` preseeds consent — so the install never hangs.

    Returns:
        ``0`` on success; ``1`` if a provisioning step (install, pull, or config
        write) fails.
    """
    try:
        result = provision(
            interactive=_interactive(), assume_yes=args.yes, pin=args.variant
        )
    except (OSError, subprocess.CalledProcessError, ConfigError) as exc:
        print(f"tux: provisioning failed: {exc}", file=sys.stderr)
        return 1
    _print_provision_result(result)
    return 0


def _print_provision_result(result: ProvisionResult) -> None:
    """Print a human summary of a provisioning run."""
    if result.bypassed:
        print(
            "tux is already pointed at a configured endpoint "
            f"({result.endpoint}); skipping provisioning."
        )
        return
    if result.decision is not None:
        print(f"Selected the {result.tier.capability} tier ({result.variant}):")
        for reason in result.decision.reasons:
            print(f"  - {reason}")
    if result.ollama_installed:
        print("Installed the Ollama runtime.")
    if result.model_pulled:
        print(f"Pulled model {result.model}.")
    elif result.model_deferred:
        print(
            f"Deferred the download of {result.model} to first run "
            "(no consent given yet)."
        )
    else:
        print(f"Model {result.model} already present.")
    if result.endpoint_reachable is False:
        print(f"warning: endpoint {result.endpoint} is not reachable yet.")
    print(f"Config now points at {result.endpoint} (model {result.model}).")


if __name__ == "__main__":
    raise SystemExit(main())
