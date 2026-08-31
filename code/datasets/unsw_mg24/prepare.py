"""Prepare UNSW-MG24 raw CICFlowMeter CSVs for the framework.

The UNSW-MG24 release ships per-attack CICFlowMeter `*.pcap_Flow.csv`
files plus department-level benign / synthetic-benign flow CSVs. The
framework expects a single ``training-flow.csv`` and ``test-flow.csv``
with column names matching the registered flow features (lowercased
snake_case from `features/flow/`) and three label columns
``attack_flag`` (binary), ``attack_step`` (kill-chain stage), and
``attack_name`` (specific attack family).

This script:

1. Walks the four UNSW-MG24 traffic directories, classifying each CSV by
   filename into a benign or specific-attack bucket.
2. Maps CICFlowMeter columns onto framework feature names.
3. Decomposes the numeric Protocol column into the framework's
   ``is_tcp`` / ``is_udp`` / ``is_icmp`` indicators and a copy as
   ``flow_protocol`` (matching other datasets).
4. Optionally subsamples each bucket so the resulting CSVs are small
   enough to train all algorithms in a few minutes.
5. Stratified-splits each bucket into train/test (default 70/30).
6. Writes ``training-flow.csv`` and ``test-flow.csv`` next to this
   script with the framework's required column order.

Run from the dataset directory:
    python prepare.py --target . [--max-per-bucket 5000]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

# CICFlowMeter -> framework feature mapping. Keys are the CSV column
# names produced by CICFlowMeter v4 (the version UNSW-MG24 ships with).
CIC_TO_FRAMEWORK: Dict[str, str] = {
    "Tot Fwd Pkts": "total_forward_packets",
    "Tot Bwd Pkts": "total_backward_packets",
    "TotLen Fwd Pkts": "total_length_of_forward_packets",
    "TotLen Bwd Pkts": "total_length_of_backward_packets",
    "Fwd Pkt Len Max": "forward_packet_length_max",
    "Fwd Pkt Len Min": "forward_packet_length_min",
    "Fwd Pkt Len Mean": "forward_packet_length_mean",
    "Fwd Pkt Len Std": "forward_packet_length_std",
    "Bwd Pkt Len Max": "backward_packet_length_max",
    "Bwd Pkt Len Min": "backward_packet_length_min",
    "Bwd Pkt Len Mean": "backward_packet_length_mean",
    "Bwd Pkt Len Std": "backward_packet_length_std",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Tot": "forward_iat_total",
    "Fwd IAT Mean": "forward_iat_mean",
    "Fwd IAT Std": "forward_iat_std",
    "Fwd IAT Max": "forward_iat_max",
    "Fwd IAT Min": "forward_iat_min",
    "Bwd IAT Tot": "backward_iat_total",
    "Bwd IAT Mean": "backward_iat_mean",
    "Bwd IAT Std": "backward_iat_std",
    "Bwd IAT Max": "backward_iat_max",
    "Bwd IAT Min": "backward_iat_min",
    "Fwd Header Len": "total_fhlen",
    "Bwd Header Len": "total_bhlen",
    "Fwd Pkts/s": "fpkts_per_second",
    "Bwd Pkts/s": "bpkts_per_second",
    "Flow Pkts/s": "flow_packets_per_second",
    "FIN Flag Cnt": "flow_fin",
    "SYN Flag Cnt": "flow_syn",
    "RST Flag Cnt": "flow_rst",
    "PSH Flag Cnt": "flow_psh",
    "ACK Flag Cnt": "flow_ack",
    "URG Flag Cnt": "flow_urg",
    "CWE Flag Count": "flow_cwr",
    "ECE Flag Cnt": "flow_ece",
}

# Always-present framework columns the model manager looks up.
FRAMEWORK_META_COLUMNS = [
    "window_start", "window_end", "protocol",
    "src_ip_addr", "src_port", "dst_ip_addr", "dst_port",
    "is_tcp", "is_udp", "is_icmp", "flow_protocol",
]

LABEL_COLUMNS = ["attack_flag", "attack_step", "attack_name"]

# Map filename keywords to (attack_name, attack_step). Synthetic benign
# files all carry the *.pcap_Flow.csv suffix; we tag them by the parent
# directory.
ATTACK_LABELS: List[Tuple[str, str, str]] = [
    # filename substring  attack_name           attack_step
    ("backdoor",          "backdoor",           "infection"),
    ("ransomware",        "ransomware",         "action"),
    ("ddos",              "ddos",               "action"),
    ("dos",               "dos",                "action"),
    ("scan1_nmap",        "nmap_scan",          "reconnaissance"),
    ("scan2_nikto",       "nikto_scan",         "reconnaissance"),
    ("hydra_password",    "password_brute",     "infection"),
    ("ftp_password",      "ftp_brute",          "infection"),
    ("sql_injection",     "sql_injection",      "infection"),
    ("shellshock",        "shellshock",         "infection"),
    ("samba_permission",  "samba_pivot",        "lateral-movement"),
    ("mitm",              "mitm",               "lateral-movement"),
    ("pivot",             "pivot",              "lateral-movement"),
    ("ms17_010",          "eternal_blue",       "infection"),
    ("meterpreter",       "meterpreter",        "persistence"),
]


def _label_for(fname: str) -> Tuple[str, int, str]:
    """Return (attack_name, attack_flag, attack_step) for a malicious CSV."""
    name = fname.lower()
    for key, aname, step in ATTACK_LABELS:
        if key in name:
            return aname, 1, step
    # Unknown malicious file -- still flag as attack but step=unknown.
    return "unknown_malicious", 1, "unknown"


def _benign_label() -> Tuple[str, int, str]:
    return "benign", 0, "benign"


def _walk_csvs(root: str) -> Iterable[Tuple[str, bool]]:
    """Yield (path, is_malicious) for every relevant CSV under root."""
    for cur, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = os.path.join(cur, f)
            rel = os.path.relpath(path, root).lower()
            # Skip everything outside the network-traffic directories.
            if "network traffic" not in rel:
                continue
            # Encryption / power / system-call CSVs are out of scope.
            if rel.startswith("encryption") or rel.startswith("power"):
                continue
            is_mal = rel.startswith("malicious network traffic")
            yield path, is_mal


def _coerce_float(v: str) -> float:
    if v is None:
        return 0.0
    s = v.strip()
    if not s:
        return 0.0
    if s.lower() in ("nan", "infinity", "-infinity"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _convert_row(
    src: Dict[str, str],
    attack_name: str,
    attack_flag: int,
    attack_step: str,
) -> Dict[str, str]:
    """Translate a CICFlowMeter row into framework column names."""
    proto = _coerce_float(src.get("Protocol", "0"))
    is_tcp = 1 if int(proto) == 6 else 0
    is_udp = 1 if int(proto) == 17 else 0
    is_icmp = 1 if int(proto) == 1 else 0

    out: Dict[str, str] = {}
    # Meta columns.
    out["window_start"] = "0"
    out["window_end"] = "0"
    out["protocol"] = str(int(proto)) if proto else "0"
    out["src_ip_addr"] = src.get("Src IP", "0.0.0.0")
    out["src_port"] = str(int(_coerce_float(src.get("Src Port", "0"))))
    out["dst_ip_addr"] = src.get("Dst IP", "0.0.0.0")
    out["dst_port"] = str(int(_coerce_float(src.get("Dst Port", "0"))))
    out["is_tcp"] = str(is_tcp)
    out["is_udp"] = str(is_udp)
    out["is_icmp"] = str(is_icmp)
    out["flow_protocol"] = str(int(proto)) if proto else "0"

    # Numeric features.
    for cic_col, fw_col in CIC_TO_FRAMEWORK.items():
        out[fw_col] = "{:.6f}".format(_coerce_float(src.get(cic_col, "0")))

    # Labels.
    out["attack_name"] = attack_name
    out["attack_flag"] = str(attack_flag)
    out["attack_step"] = attack_step
    return out


def _gather(root: str, max_per_bucket: Optional[int]) -> List[Dict[str, str]]:
    """Read every relevant CSV, optionally subsample, return rows."""
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for path, is_mal in _walk_csvs(root):
        if is_mal:
            aname, aflag, astep = _label_for(os.path.basename(path))
            bucket_key = aname
        else:
            aname, aflag, astep = _benign_label()
            bucket_key = "benign"
        bucket = buckets.setdefault(bucket_key, [])
        try:
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    bucket.append(_convert_row(row, aname, aflag, astep))
        except Exception as exc:  # pragma: no cover -- defensive
            logging.warning("skip %s: %s", path, exc)

    rows: List[Dict[str, str]] = []
    for key, items in buckets.items():
        if max_per_bucket is not None and len(items) > max_per_bucket:
            random.shuffle(items)
            items = items[:max_per_bucket]
        logging.info("bucket %s -> %d rows", key, len(items))
        rows.extend(items)
    random.shuffle(rows)
    return rows


def _write_split(rows: List[Dict[str, str]], target: str, train_ratio: float) -> None:
    if not rows:
        raise RuntimeError("No rows produced. Check raw CSVs are present.")
    columns = (
        FRAMEWORK_META_COLUMNS
        + sorted(CIC_TO_FRAMEWORK.values())
        + LABEL_COLUMNS
    )
    train_path = os.path.join(target, "training-flow.csv")
    test_path = os.path.join(target, "test-flow.csv")
    n_train = int(len(rows) * train_ratio)
    train_rows, test_rows = rows[:n_train], rows[n_train:]

    for path, subset in ((train_path, train_rows), (test_path, test_rows)):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
        logging.info("wrote %s (%d rows)", path, len(subset))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--target", default=".",
                   help="UNSW-MG24 root containing the per-modality directories.")
    p.add_argument("--max-per-bucket", type=int, default=5000,
                   help="Cap rows per attack family / benign bucket. "
                        "Use 0 to keep everything.")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-l", "--log", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log)
    random.seed(args.seed)

    root = os.path.abspath(args.target)
    if not os.path.isdir(root):
        logging.error("target dir does not exist: %s", root)
        sys.exit(1)

    cap = None if args.max_per_bucket == 0 else args.max_per_bucket
    rows = _gather(root, cap)
    _write_split(rows, root, args.train_ratio)


if __name__ == "__main__":
    main()
