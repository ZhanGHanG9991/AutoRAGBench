"""Run a command inside a network-less sandbox and bridge it to the host over unix sockets.

  relay.py --sock-dir D --port P --llm-port L -- bash submission/start.sh

  host evaluator  --(D/service.sock)-->  relay  --(127.0.0.1:P)-->  service
  service  --(127.0.0.1:L)-->  relay  --(D/gateway.sock)-->  host gateway

The command inherits the environment (LLM_BASE_URL should point at 127.0.0.1:L). Exits with the command.
"""
import argparse
import asyncio
import os
import signal
import sys


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def bridge(connect):
    async def handle(r1, w1):
        try:
            r2, w2 = await connect()
        except OSError:
            w1.close()
            return
        await asyncio.gather(pipe(r1, w2), pipe(r2, w1))
    return handle


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--llm-port", type=int, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    svc_sock = os.path.join(a.sock_dir, "service.sock")
    if os.path.exists(svc_sock):
        os.remove(svc_sock)
    s1 = await asyncio.start_unix_server(bridge(lambda: asyncio.open_connection("127.0.0.1", a.port)), svc_sock)
    s2 = await asyncio.start_server(
        bridge(lambda: asyncio.open_unix_connection(os.path.join(a.sock_dir, "gateway.sock"))), "127.0.0.1", a.llm_port)
    proc = await asyncio.create_subprocess_exec(*cmd, start_new_session=True)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: os.killpg(proc.pid, signal.SIGTERM))
    code = await proc.wait()
    s1.close()
    s2.close()
    sys.exit(code)


if __name__ == "__main__":
    asyncio.run(main())
