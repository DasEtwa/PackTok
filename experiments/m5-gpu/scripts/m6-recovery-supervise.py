"""Single-allocation M6 probe. Run in dedicated Ubuntu-24.04 after the CPU rehearsal."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid


def inventory(text):
    rows = re.findall(r"^\[([^]]+)\] (\S+) \| Hardware: ([^ |\n]+)", text, re.M)
    if not rows and "No active sessions found on server." not in text:
        raise RuntimeError("unrecognized inventory")
    return rows


def usage(text):
    balance = re.search(r"^Current balance: ([0-9.]+) compute units$", text, re.M)
    rate = re.search(r"^Usage rate: ([0-9.]+)/hr$", text, re.M)
    count = re.search(r"^Active assignments: (\d+)$", text, re.M)
    if not all([balance, rate, count]):
        raise RuntimeError("unrecognized CU accounting")
    return float(balance[1]), float(rate[1]), int(count[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("package", type=Path)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--cu", type=float, required=True)
    a = p.parse_args()
    if not 180 < a.seconds <= 1200 or not 0 < a.cu <= 0.60:
        p.error("outside single-probe budget")
    package = a.package.resolve()
    ready = json.loads((package / "ready.json").read_text())
    bridge = Path(__file__).with_name("m6-recovery-remote.py")
    def sha(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    if sha(bridge) != ready["bridge_sha256"] or sha(package/"bundle.tar.gz") != ready["bundle_sha256"]:
        raise ValueError("unverified package or bridge")
    parts = sorted(package.glob("m6-probe.part???"))
    if [sha(i) for i in parts] != ready["part_sha256"] or not parts or ready["cpu_rehearsal"] != "PASS":
        raise ValueError("unverified transfer parts or CPU rehearsal")
    run = package / "allocation"
    run.mkdir(mode=0o700)  # durable single-attempt marker, cannot be reused
    alias = "packtok-m6-" + uuid.uuid4().hex[:12]
    deadline = None
    def call(label, args, seconds):
        limit = seconds if deadline is None else min(seconds, deadline - time.monotonic())
        if limit <= 0:
            raise TimeoutError(label)
        start = time.monotonic()
        rc = None
        with (run / (label + ".txt")).open("xb") as out:
            try:
                r = subprocess.run(["colab", *args], stdout=out, stderr=subprocess.STDOUT, timeout=limit)
                rc = r.returncode
                if rc:
                    raise RuntimeError(f"{label} CLI exit {rc}")
            finally:
                with (run / "stages.jsonl").open("a") as log:
                    print(json.dumps(dict(stage=label, seconds=time.monotonic()-start, exit=rc)), file=log)
        return (run / (label + ".txt")).read_text()
    if inventory(call("sessions-before", ["sessions"], 25)):
        raise RuntimeError("allocation requires an empty initial inventory")
    balance, rate, count = usage(call("usage-before", ["usage"], 25))
    if count or rate or balance < a.cu:
        raise RuntimeError("usage precondition failed")
    endpoint = None
    rc = 1
    released = False
    active_rate = None
    ending_balance = None
    start = time.monotonic()
    start_epoch = time.time()
    deadline = start + a.seconds - 130
    (run/"request.json").write_text(json.dumps(dict(alias=alias,utc_epoch=start_epoch,cap_seconds=a.seconds,cap_cu=a.cu)))
    def interrupt(*_):
        raise InterruptedError("supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        call("allocation", ["new", "-s", alias, "--gpu", "L4"], 150)
        live = inventory(call("sessions-active", ["sessions"], 20))
        own = [row for row in live if row[0] == alias]
        if len(own) != 1:
            raise RuntimeError("owned endpoint missing")
        _, endpoint, hardware = own[0]
        if hardware != "L4":
            raise RuntimeError("wrong accelerator")
        _, active_rate, count = usage(call("usage-active", ["usage"], 25))
        if count != 1 or active_rate <= 0 or active_rate*a.seconds/3600 > a.cu:
            raise RuntimeError("active rate outside budget")
        for i, part in enumerate(parts):
            call(f"upload-{i:03d}", ["upload","-s",alias,str(part),"content/"+part.name],100)
        call("execute", ["exec","-s",alias,"-f",str(bridge),"--timeout","270","--env",
                         "PACKTOK_M6_SHA256="+ready["bundle_sha256"]],285)
        call("download", ["download","-s",alias,"content/m6-result.tar.gz",str(run/"results.tar.gz")],40)
        rc = 0
    except BaseException as e:
        (run/"failure.txt").write_text(str(e)+"\n")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        deadline = start + a.seconds
        for attempt in range(2):
            try:
                live = inventory(call(f"before-stop-{attempt}",["sessions"],15))
                own = [row for row in live if row[0] == alias]
                if not own:
                    if endpoint is None and live:
                        raise RuntimeError("unidentified assignment after request")
                    if any(row[1] == endpoint for row in live):
                        raise RuntimeError("owned endpoint has been reassigned")
                    break
                if len(own) != 1 or (endpoint is not None and endpoint != own[0][1]):
                    raise RuntimeError("ownership changed")
                # Unique per-request alias was absent before this one allocation; confirm its mapping.
                status = inventory(call(f"status-{attempt}",["status","-s",alias],10))
                if status != own:
                    raise RuntimeError("status and server inventory disagree")
                endpoint = own[0][1]
                call(f"stop-{attempt}",["stop","-s",alias],20)
                break
            except Exception as e:
                (run/f"release-error-{attempt}.txt").write_text(str(e)+"\n")
        try:
            rows = inventory(call("sessions-after",["sessions"],18))
            released = not any(r[0]==alias or r[1]==endpoint for r in rows) and (endpoint is not None or not rows)
            ending_balance, _, _ = usage(call("usage-after",["usage"],18))
        finally:
            (run/"accounting.json").write_text(json.dumps(dict(allocation_requests=1,release_verified=released,
                seconds=time.monotonic()-start,cu_before=balance,cu_after=ending_balance,active_rate=active_rate,
                transport_exit=rc),indent=2)+"\n")
    return 0 if released and rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
