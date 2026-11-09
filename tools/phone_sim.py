#!/usr/bin/env python3
"""Phone-night simulator: drives `rust_trainer run` through a realistic night.

It plays the part of agentd's device monitor. It launches the trainer, measures
the trainer's real CPU use from /proc, feeds that into a simple phone thermal
model, simulates charging/battery and the Light→Deep stage, injects an unplug
event, and writes the status file the trainer reads before every example:

    {"stage":"deep","charging":true,"battery_pct":83,"temp_c":31.5}

Thermal model (per second):  T += heat * cpu_util - cool * (T - ambient)
With the defaults, sustained full load settles around 41 °C, above the 40 °C
limit, so the trainer has to pause and resume as it would on a hot phone.
Time is compressed: 1 s here stands for a little more on a phone.

usage: phone_sim.py --status dev.json --log night.csv [--unplug-at 240 --unplug-for 45]
                    [--deep-after 15] [--battery 46] -- rust_trainer run --status dev.json ...
"""
import argparse, json, os, subprocess, sys, time

def cpu_seconds(pid):
    """utime+stime of the process (all threads) in seconds."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except FileNotFoundError:
        return None

def rss_peak_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return None

def write_state(path, st):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    ap.add_argument("--log", default="night.csv")
    ap.add_argument("--cores", type=float, default=os.cpu_count(), help="cores the trainer may use (for utilisation)")
    ap.add_argument("--ambient", type=float, default=28.0)
    ap.add_argument("--start-temp", type=float, default=30.0)
    ap.add_argument("--heat", type=float, default=0.26, help="°C per second at 100%% CPU")
    ap.add_argument("--cool", type=float, default=0.02, help="fraction of excess heat lost per second")
    ap.add_argument("--battery", type=float, default=46.0)
    ap.add_argument("--charge-rate", type=float, default=0.25, help="%% per second while plugged")
    ap.add_argument("--drain-rate", type=float, default=0.05, help="%% per second while unplugged and busy")
    ap.add_argument("--deep-after", type=float, default=15.0)
    ap.add_argument("--unplug-at", type=float, default=-1)
    ap.add_argument("--unplug-for", type=float, default=45.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("give the trainer command after --")

    temp, batt, t0 = a.start_temp, a.battery, time.time()
    state = {"stage": "light", "charging": True, "battery_pct": round(batt, 1), "temp_c": round(temp, 1)}
    write_state(a.status, state)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    last_cpu, last_t = 0.0, time.time()
    log = open(a.log, "w")
    log.write("t,stage,charging,battery_pct,temp_c,cpu_util,phase,step\n")
    max_temp, busy_s, rows, peak = temp, 0.0, 0, 0.0
    while proc.poll() is None:
        time.sleep(1.0)
        now = time.time()
        t = now - t0
        cpu = cpu_seconds(proc.pid)
        if cpu is None:
            break
        util = max(0.0, min(1.0, (cpu - last_cpu) / max(1e-6, (now - last_t) * a.cores)))
        last_cpu, last_t = cpu, now
        busy_s += util
        unplugged = a.unplug_at >= 0 and a.unplug_at <= t < a.unplug_at + a.unplug_for
        charging = not unplugged
        stage = "active" if unplugged else ("deep" if t >= a.deep_after else "light")
        temp += a.heat * util - a.cool * (temp - a.ambient)
        batt = min(100.0, batt + a.charge_rate) if charging else max(0.0, batt - a.drain_rate * (0.2 + util))
        max_temp = max(max_temp, temp)
        state = {"stage": stage, "charging": charging, "battery_pct": round(batt, 1), "temp_c": round(temp, 2)}
        write_state(a.status, state)
        phase, step = "", ""
        try:
            for root, _, files in os.walk(os.path.dirname(os.path.abspath(a.status))):
                if "status.json" in files and root.endswith("work"):
                    s = json.load(open(os.path.join(root, "status.json")))
                    phase, step = s.get("phase", ""), f'{s.get("step", "")}/{s.get("total_steps", "")}'
                    break
        except Exception:
            pass
        log.write(f"{t:.0f},{stage},{int(charging)},{batt:.1f},{temp:.2f},{util:.2f},{phase},{step}\n")
        log.flush()
        rows += 1
        peak = max(peak, rss_peak_mb(proc.pid) or 0.0)
    out = proc.stdout.read()
    elapsed = time.time() - t0
    print(out, end="")
    print(json.dumps({
        "sim_seconds": round(elapsed),
        "max_temp_c": round(max_temp, 1),
        "avg_cpu_util": round(busy_s / max(1, rows), 2),
        "peak_rss_mb": round(peak),
        "exit_code": proc.returncode,
    }), file=sys.stderr)

if __name__ == "__main__":
    main()
