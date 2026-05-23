"""Export the synthetic Starlink-shell satellites to a checked-in 3-line TLE
file (data/tle/starlink_synth.tle) so the *exact same* satellites can be
reproduced from disk in the absence of internet access to Celestrak.

The TLE strings are produced by sgp4.exporter.export_tle, so they round-trip
through skyfield/sgp4 exactly. Re-run this script whenever
:func:`satqkd.orbit_tle.starlink_shell` defaults change.

Usage:
  python scripts/make_synthetic_tle.py            # default n=20 shell
  python scripts/make_synthetic_tle.py --n 30 --out data/tle/large.tle
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd.orbit_tle import starlink_shell  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="number of satellites")
    ap.add_argument("--altitude-km", type=float, default=550.0)
    ap.add_argument("--epoch", type=str, default="2026-05-23T12:00:00",
                    help="UTC epoch (ISO8601) — the SGP4 epoch_days field.")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parents[1]
                                / "data" / "tle" / "starlink_synth.tle"))
    args = ap.parse_args()

    from sgp4.exporter import export_tle
    epoch_utc = datetime.fromisoformat(args.epoch).replace(tzinfo=timezone.utc)
    sats = starlink_shell(n_sats=args.n, altitude_km=args.altitude_km,
                          epoch_utc=epoch_utc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Synthetic Starlink-shell TLE bundle ({args.n} sats, "
                 f"{args.altitude_km:.0f} km, 53 deg incl.)")
    lines.append(f"# epoch: {args.epoch} UTC")
    lines.append(f"# regenerate: python scripts/make_synthetic_tle.py "
                 f"--n {args.n} --epoch {args.epoch}")
    lines.append("")
    for name, sat in sats:
        l1, l2 = export_tle(sat)
        lines.append(name)
        lines.append(l1)
        lines.append(l2)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.n} sats -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
