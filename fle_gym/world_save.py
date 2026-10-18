"""Request and export a real Factorio save archive from a connected environment.

The server may have unserialisable Lua functions in `storage`.  The validated
project save method recursively removes such function references before saving.
This is intentionally opt-in; use after a rollout, before closing/resetting.
Docker is used only for copying the generated .zip from the running container.
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any


class WorldSaveError(RuntimeError):
    """Server save request or archive export failed."""


def _clean_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:120]
    if not name or name in {".", ".."}:
        raise ValueError("Save name must contain a letter or digit")
    return name


def _run(*cmd: str, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise WorldSaveError("Docker CLI not available. The Factorio save may exist inside the container, "
                             "but export to the host requires docker or a mounted saves folder.") from exc


def _docker_for_port(port: int) -> str:
    """Choose the running container publishing the specified host TCP port."""
    p = _run("docker", "ps", "--format", "{{.ID}}", timeout=20)
    if p.returncode:
        raise WorldSaveError("docker ps failed: " + p.stderr.strip())
    ids = p.stdout.split()
    if not ids:
        raise WorldSaveError("No running Docker containers available for save export")
    p = _run("docker", "inspect", *ids, timeout=20)
    if p.returncode:
        raise WorldSaveError("docker inspect failed: " + p.stderr.strip())
    matches = []
    for item in json.loads(p.stdout):
        ports = item.get("NetworkSettings", {}).get("Ports") or {}
        for endpoint, bindings in ports.items():
            if not endpoint.endswith("/tcp") or not bindings:
                continue
            if any(int(b.get("HostPort", -1)) == int(port) for b in bindings):
                matches.append(item["Id"])
                break
    if len(matches) != 1:
        raise WorldSaveError(f"Expected one container mapping host TCP port {port}, found {len(matches)}. "
                             "Provide container= explicitly to save_world().")
    return matches[0]


def _connected_instance(env: Any):
    """Find the already-connected FLE FactorioInstance across wrapper layers."""
    pending = [env]
    visited = set()
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in visited:
            continue
        visited.add(id(obj))
        try:
            client = getattr(obj, "rcon_client", None)
            if client is not None and callable(getattr(client, "send_command", None)):
                return obj
        except Exception:
            pass
        for key in ("instance", "env", "_env", "base_env", "_base_env", "unwrapped", "wrapped_env"):
            try:
                child = getattr(obj, key, None)
            except Exception:
                continue
            if child is not None and child is not obj:
                pending.append(child)
        if len(visited) > 80:
            break
    raise WorldSaveError("Could not locate the connected FLE instance in this wrapper. "
                         "Call save_world while the environment is open, before env.close().")


def _request_save(rcon_client: Any, name: str, scrub_storage: bool) -> dict[str, Any]:
    marker = "FLE_GYM_SAVE_OK|"
    scrub = (
        'local seen={}; local removed=0; '
        'local function scrub(t) if type(t)~="table" or seen[t] then return end; seen[t]=true; '
        'local drop={}; for k,v in pairs(t) do '
        'if type(k)=="function" or type(v)=="function" then '
        'drop[#drop+1]=k; removed=removed+1 '
        'elseif type(v)=="table" then scrub(v) end end; '
        'for _,k in ipairs(drop) do t[k]=nil end end; '
        'scrub(storage); '
    ) if scrub_storage else 'local removed=0; '
    lua = (
        '/sc ' + scrub +
        'local ok,err=pcall(function() game.server_save(' + json.dumps(name) + ') end); '
        'local api="server_save"; '
        'if not ok then game.autosave_enabled=true; '
        'ok,err=pcall(function() game.auto_save(' + json.dumps(name) + ',true) end); '
        'api="auto_save_two_arg" end; '
        'if not ok then ok,err=pcall(function() game.auto_save(' + json.dumps(name) + ') end); '
        'api="auto_save_one_arg" end; '
        'if not ok then error(err) end; '
        'rcon.print(' + json.dumps(marker) + '..' + json.dumps(name) + '.."|api="..api.."|removed="..removed)'
    )
    response = str(rcon_client.send_command(lua) or "")
    if marker not in response or "cannot execute command" in response.lower() or "filesystem error" in response.lower():
        raise WorldSaveError("Factorio did not confirm world save: " + response[-1000:])
    suffix = response[response.index(marker) + len(marker):]
    api = suffix.split("|api=", 1)[-1].split("|", 1)[0]
    return {"server_response": response, "method": api, "save_name": name,
            "lua_storage_scrubbed": bool(scrub_storage)}


def save_factorio_world(env: Any, name: str, output_path: str | Path,
                        *, container: str | None = None, rcon_port: int | None = None,
                        scrub_storage: bool = True, wait_seconds: float = 30.0) -> Path:
    """Create and export a verified Factorio save .zip to *output_path*.

    Must be called on an open, connected environment, preferably after its last
    step: scrubbing unserialisable Lua state is deliberately destructive to
    Lua function references. Requires Docker CLI access, or specify the running
    container id/name explicitly. Raises WorldSaveError instead of declaring a
    success merely because Factorio acknowledged an RCON command.
    """
    save_name = _clean_name(name)
    out = Path(output_path).expanduser().resolve()
    if out.suffix.lower() != ".zip":
        raise ValueError("output_path must end in .zip (a real Factorio save archive)")
    if out.exists():
        raise FileExistsError(f"Will not overwrite an existing world save: {out}")
    instance = _connected_instance(env)
    if rcon_port is None:
        for key in ("rcon_port", "tcp_port", "port"):
            value = getattr(env, key, None)
            if isinstance(value, int):
                rcon_port = value
                break
    if container is None:
        if rcon_port is None:
            raise WorldSaveError("Specify rcon_port= or container= to locate the Factorio container")
        container = _docker_for_port(rcon_port)

    # Scrub and save only after establishing how to export; avoid modifying
    # the live Lua state if Docker is unavailable.
    result = _request_save(instance.rcon_client, save_name, scrub_storage)
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    candidates = [f"{save_name}.zip", f"_autosave-{save_name}.zip"]
    checked = []
    while True:
        for directory in ("/factorio/saves", "/opt/factorio/saves"):
            for candidate in candidates:
                remote = f"{container}:{directory}/{candidate}"
                part = out.with_name(out.name + ".partial")
                p = _run("docker", "cp", remote, str(part), timeout=30)
                checked.append(remote)
                if p.returncode != 0:
                    part.unlink(missing_ok=True)
                    continue
                try:
                    with zipfile.ZipFile(part, "r") as archive:
                        if not archive.namelist() or archive.testzip() is not None:
                            raise WorldSaveError(f"Factorio save archive corrupt: {remote}")
                    if part.stat().st_size < 1024:
                        raise WorldSaveError(f"Factorio save archive unexpectedly small: {remote}")
                    part.replace(out)
                    return out
                finally:
                    part.unlink(missing_ok=True)
        if time.monotonic() >= deadline:
            raise WorldSaveError("Factorio acknowledged the save but a valid ZIP was not exported. "
                                 f"API={result['method']}; searched: {sorted(set(checked))}")
        time.sleep(1)
