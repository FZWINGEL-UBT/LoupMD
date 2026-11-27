"""Estimate Stage 0 CU from its allocation receipts, not the shared account's balance change."""
import argparse
import json
import math
import time
from pathlib import Path

# Incremental hourly CU observed on the current Colab plan. Not provider invoices.
RATE = {'T4': 1.15, 'L4': 1.94}


def tally(root, now):
    accounts = {a: {'estimated_cu': 0., 'active_sessions': [], 'no_allocation_receipt': []} for a in ('1', '2')}
    seen = set()
    for path in sorted(root.rglob('allocation.json')):
        r = json.loads(path.read_text())
        session = r['session']
        if session in seen:
            raise ValueError(f'duplicate allocation receipt: {session}')
        seen.add(session)
        account = accounts[r['account']]
        if 'allocated' not in r:
            account['no_allocation_receipt'].append(session)
            continue
        # The old controller sometimes returned after an independently verified
        # manual stop. Charge to the first real stop, not its later acknowledgement.
        stops = [r['stopped']] if 'stopped' in r else []
        for marker in ('checked/stopped-unix', 'recovery/stopped-unix'):
            p = path.parent / marker
            if p.is_file():
                stops.append(float(p.read_text()))
        start, stop = r['allocated'], min(stops) if stops else now
        if not (math.isfinite(start) and math.isfinite(stop) and start <= stop <= now):
            raise ValueError(f'invalid allocation interval for {session}: {start} to {stop}')
        account['estimated_cu'] += RATE[r['gpu']] * (stop - start) / 3600
        if not stops:
            account['active_sessions'].append(session)
    return {'as_of_unix': now, 'estimated_rates_cu_per_hour': RATE, 'accounts': accounts}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign_root', type=Path)
    root = p.parse_args().campaign_root
    if not root.is_dir():
        p.error('expected an existing Stage 0 campaign root')
    print(json.dumps(tally(root, time.time()), indent=2))
