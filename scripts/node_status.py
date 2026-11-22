"""
node_status.py
Print a plain-English sync status for the local Bitcoin node, so you know when
the ZMQ live feed is ready.

    py scripts/node_status.py

Reads the node connection settings from the environment (BATMAN_RPC_COOKIE etc.;
see src/config.get_node_settings). "READY" means the initial sync is finished
and you can run:  py -m src.core.backend.live.monitor --source zmq
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_node_settings
from src.core.backend.live.node_rpc import BitcoinRPC, NodeRPCError


def main() -> int:
    settings = get_node_settings()
    rpc = BitcoinRPC(url=settings["rpc_url"], user=settings["rpc_user"],
                     password=settings["rpc_password"], cookie_path=settings["rpc_cookie"])
    try:
        info = rpc.call("getblockchaininfo")
    except NodeRPCError as error:
        print("Node not reachable:", error)
        print("Is it running? Start it with:")
        print(r'  "D:\BITCOIN\daemon\bitcoind.exe" -datadir=D:/BitcoinData -conf=D:/BitcoinData/bitcoin.conf')
        return 1

    syncing = info.get("initialblockdownload", True)
    progress = info.get("verificationprogress", 0.0) * 100
    bar_len = 30
    filled = int(bar_len * progress / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    print(f"  chain        : {info['chain']}  (pruned: {info.get('pruned')})")
    print(f"  blocks       : {info['blocks']:,} of ~{info['headers']:,}")
    print(f"  size on disk : {info.get('size_on_disk', 0) / 1e9:.2f} GB")
    print(f"  progress     : [{bar}] {progress:.2f}%")
    if syncing:
        print("  STATUS       : STILL SYNCING -- keep using the Public API feed for now.")
    else:
        print("  STATUS       : READY  ->  py -m src.core.backend.live.monitor --source zmq")
    return 0


if __name__ == "__main__":
    sys.exit(main())
