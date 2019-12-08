# readline proxy, intended to be run in a subprocess.
#
# Set it up with a terminal (could be a pseudoterminal, or could just
# be a dumb pipe but then you don't get line editing/etc) connected to
# stdin and stdout, and pass as sole command-line argument the path
# of a UNIX listening socket (the "control socket"). Traffic over
# the control socket is newline-delimited JSON encoded lists
# of strings. rlproxy's main loop interacts with the control
# socket as follows:
#
# - Handling phase: Repeatedly read and process control socket input:
#   - ["output", str]: write str to stdout and flush it
#   - ["prompt", str]: set our input prompt to str (initial default is empty)
#   - ["done"]: exit the handling phase, proceed to read a command from stdin
#   - ["quit"]: exit rlproxy entirely (this also occurs if we read EOF)
#
# - Prompt phase: Read a command from stdin using readline. If tab
#   completions are needed, send ["complete", line, word, begidx] over
#   the control socket, where line is the entire input line so far and
#   word is the thing readline is trying to complete, which starts at
#   offset begidx in line.  Read a single response to each completion
#   request over the control socket; the response should be a list
#   (possibly empty) of potential completions.
#
# - Send ["run", line-the-user-typed] over the control socket and
#   enter another handling phase to process the responses.  If the
#   user entered EOF, send ["eof"] instead of ["run", ...] and still
#   enter the handling phase.  If a Ctrl+C is received while command
#   is running, send ["interrupt"] to the remote side.

import json
import readline
import signal
import socket
import sys
from typing import Any, List, NoReturn, Optional, cast


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m rlproxy /path/to/control.sock")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sys.argv[1])
    control_in = sock.makefile("r", buffering=1, encoding="utf-8")
    control_out = sock.makefile("w", buffering=1, encoding="utf-8")

    def send_control(obj: List[str]) -> None:
        control_out.write(json.dumps(obj) + "\n")
        control_out.flush()

    def recv_control() -> List[str]:
        line = control_in.readline()
        if not line:
            sys.exit(0)
        return cast(List[str], json.loads(line.rstrip("\n")))

    def completer(text: str, state: int, _last: List[str] = []) -> Optional[str]:
        if state == 0:
            send_control(
                [
                    "complete",
                    readline.get_line_buffer(),
                    text,
                    str(readline.get_begidx()),
                ]
            )
            _last[:] = recv_control()
        try:
            return _last[state]
        except IndexError:
            return None

    prompt = ""
    interrupt_pending = False
    waiting_on_input = False

    def sigint_handler(*args: Any) -> None:
        nonlocal interrupt_pending
        if waiting_on_input:
            send_control(["interrupt"])
            interrupt_pending = False
        else:
            interrupt_pending = True

    def sigterm_handler(*args: Any) -> NoReturn:
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    def forward_response() -> None:
        nonlocal interrupt_pending, waiting_on_input
        interrupt_pending = False
        while True:
            waiting_on_input = True
            try:
                if interrupt_pending:
                    sigint_handler()
                line = recv_control()
            finally:
                waiting_on_input = False

            if line[0] == "output":
                sys.stdout.write(line[1])
                sys.stdout.flush()
            elif line[0] == "prompt":
                nonlocal prompt
                prompt = line[1]
            elif line[0] == "done":
                break
            elif line[0] == "quit":
                sys.exit(0)
            else:
                sys.exit(f"invalid input from control process: {line!r}")

    readline.parse_and_bind("tab: complete")
    readline.set_completer(completer)
    while True:
        forward_response()
        try:
            cmd = input(prompt)
        except EOFError:
            sys.stdout.write("\n")
            sys.stdout.flush()
            send_control(["eof"])
        else:
            send_control(["run", cmd.rstrip()])


if __name__ == "__main__":
    main()
