#!/usr/bin/env python3
"""Validate a chunked orange recording (a session folder with chunk_NNNNN/).

Usage:  python3 tools/check_chunks.py /mnt/data/exp/unsorted/<session> [--ffprobe PATH] [--no-video]

Checks, per camera and across cameras:
  continuity   frame_id has no gaps across all chunks (gap = encoder queue full)
               and the PTP timestamp never jumps by more than 1.5 periods
               (jump = camera/NIC drop), including across every chunk boundary
  alignment    every camera's chunk k starts on the same exposure: the first
               timestamps of chunk k agree to well under one frame period, and
               every camera has the same number of frames in chunk k
  video        each Cam*.mp4 has as many packets as its _meta.csv has rows and
               its first packet is a keyframe (needs ffprobe; skip with --no-video)
  complete     each chunk folder has chunk_complete.json (written once every
               camera has closed its files for that chunk)

Exit status 1 if any check fails.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def read_meta(path):
    rows = []
    with open(path, "r", buffering=1 << 20) as f:
        f.readline()
        for line in f:
            p = line.split(",")
            if len(p) >= 2:
                try:
                    rows.append((int(p[0]), int(p[1])))
                except ValueError:
                    pass
    return rows


def ffprobe_env(ffprobe):
    # A locally built ffprobe (e.g. ~/nvidia/ffmpeg/build/bin) needs its own
    # shared libraries from ../lib.
    env = dict(os.environ)
    lib = os.path.join(os.path.dirname(os.path.realpath(ffprobe)), "..", "lib")
    if os.path.isdir(lib):
        env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def probe(ffprobe, mp4):
    """(packet count, first packet is key) of the video stream."""
    env = ffprobe_env(ffprobe)
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", mp4],
        capture_output=True, text=True, env=env)
    if out.returncode != 0:
        sys.exit("ffprobe failed on %s: %s\n(use --ffprobe PATH or --no-video)"
                 % (mp4, out.stderr.strip()[:300]))
    try:
        n = int(out.stdout.strip().split(",")[0])
    except ValueError:
        n = -1
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals",
         "%+#1", "-show_entries", "packet=flags", "-of", "csv=p=0", mp4],
        capture_output=True, text=True, env=env)
    key = out.stdout.strip().startswith("K")
    return n, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--ffprobe", default=shutil.which("ffprobe") or
                    os.path.expanduser("~/nvidia/ffmpeg/build/bin/ffprobe"))
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    chunks = sorted(glob.glob(os.path.join(a.session, "chunk_*")))
    if not chunks:
        print("no chunk_* folders in", a.session)
        return 1
    serials = sorted({os.path.basename(p)[3:-9]
                      for c in chunks for p in glob.glob(os.path.join(c, "Cam*_meta.csv"))})
    video = not a.no_video and a.ffprobe and os.path.exists(a.ffprobe)
    fails = []

    # meta[serial][chunk_idx] = rows
    meta = {s: [] for s in serials}
    for c in chunks:
        for s in serials:
            p = os.path.join(c, "Cam%s_meta.csv" % s)
            meta[s].append(read_meta(p) if os.path.exists(p) else None)

    # Frame period from the first camera's first chunk (median delta).
    first = next(r for r in meta[serials[0]] if r and len(r) > 2)
    d = sorted(b[1] - a_[1] for a_, b in zip(first, first[1:]))
    period = d[len(d) // 2]
    print("%s: %d chunks, %d cameras, period %.3f ms (%.2f fps)" % (
        a.session, len(chunks), len(serials), period / 1e6, 1e9 / period))

    hdr = "chunk  " + " ".join("%9s" % s[-9:] for s in serials) + "   start spread  notes"
    print(hdr)
    for k, c in enumerate(chunks):
        notes = []
        counts, starts = [], []
        for s in serials:
            r = meta[s][k]
            if r is None:
                counts.append(-1)
                notes.append("Cam%s missing" % s)
                continue
            counts.append(len(r))
            if r:
                starts.append(r[0][1])
            if video:
                mp4 = os.path.join(c, "Cam%s.mp4" % s)
                n, key = probe(a.ffprobe, mp4)
                if n != len(r):
                    notes.append("Cam%s mp4 %d pkts vs %d rows" % (s, n, len(r)))
                if not key:
                    notes.append("Cam%s first packet not key" % s)
        spread = (max(starts) - min(starts)) if starts else 0
        last = k == len(chunks) - 1
        if len(set(counts)) != 1:
            notes.append("frame counts differ")
        if spread > period / 4:
            notes.append("cameras start on different exposures")
        if not os.path.exists(os.path.join(c, "chunk_complete.json")):
            notes.append("no chunk_complete.json")
        print("%5d  " % k + " ".join("%9d" % n for n in counts) +
              "   %9d ns  %s" % (spread, "; ".join(notes) or "ok"))
        if notes:
            fails.append("chunk %d: %s" % (k, "; ".join(notes)))

    # Continuity of each camera across the whole session.
    print()
    for s in serials:
        rows = [x for r in meta[s] if r for x in r]
        id_gaps = sum(1 for p, q in zip(rows, rows[1:]) if q[0] - p[0] != 1)
        id_missing = sum(q[0] - p[0] - 1 for p, q in zip(rows, rows[1:]) if q[0] - p[0] > 1)
        ts_jumps = [(q[1] - p[1]) for p, q in zip(rows, rows[1:]) if q[1] - p[1] > 1.5 * period]
        ts_missing = sum(round(j / period) - 1 for j in ts_jumps)
        # boundary deltas between consecutive chunks
        bd = []
        for r0, r1 in zip(meta[s], meta[s][1:]):
            if r0 and r1:
                bd.append(round((r1[0][1] - r0[-1][1]) / period, 3))
        bad_b = [x for x in bd if abs(x - 1.0) > 0.25]
        ok = id_gaps == 0 and not ts_jumps and not bad_b
        print("Cam%s: %d frames, encoder gaps %d (%d frames), camera gaps %d (%d frames), "
              "boundaries %d (all one period apart: %s) %s" % (
                  s, len(rows), id_gaps, id_missing, len(ts_jumps), ts_missing,
                  len(bd), "yes" if not bad_b else "NO " + str(bad_b[:5]),
                  "OK" if ok else "FAIL"))
        if not ok:
            fails.append("Cam%s continuity" % s)

    print()
    if fails:
        print("FAIL:", len(fails), "problem(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
