"""
test_live_bitcoind.py
Optional integration test: the ZMQ feed against a real Bitcoin Core node.

It starts a throwaway regtest bitcoind (a private chain in a temp folder,
ready in seconds, no network sync), sends real transactions, and checks the
feed end to end through build_zmq_fetch / run_cycle:

- transactions sent while no cycle is running are all processed (0 missed),
  with sending addresses resolved (Core gives no prevouts for mempool txs);
- transactions mined before their cycle runs are still found;
- bitcoind's re-announcement of mined transactions is skipped;
- a child of an unconfirmed parent resolves its inputs.

Skipped unless BATMAN_TEST_BITCOIND points to a bitcoind executable, e.g.
    set BATMAN_TEST_BITCOIND=C:\\path\\to\\bitcoin-32.1\\bin\\bitcoind.exe
"""

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

BITCOIND = os.environ.get("BATMAN_TEST_BITCOIND")
pytestmark = pytest.mark.skipif(not BITCOIND or not Path(BITCOIND).exists(),
                                reason="set BATMAN_TEST_BITCOIND to a bitcoind binary")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def node(tmp_path_factory):
    from src.core.backend.live.node_rpc import BitcoinRPC, NodeRPCError

    datadir = tmp_path_factory.mktemp("regtest")
    rpc_port, zmq_port = free_port(), free_port()
    (datadir / "bitcoin.conf").write_text(
        "regtest=1\nserver=1\nfallbackfee=0.0002\n[regtest]\n"
        f"rpcport={rpc_port}\nrpcbind=127.0.0.1\nrpcallowip=127.0.0.1\nlisten=0\n"
        f"zmqpubhashtx=tcp://127.0.0.1:{zmq_port}\n"
    )
    process = subprocess.Popen([BITCOIND, f"-datadir={datadir}", "-printtoconsole=0"])
    cookie = datadir / "regtest" / ".cookie"
    url = f"http://127.0.0.1:{rpc_port}"
    deadline = time.time() + 60
    while True:
        try:
            BitcoinRPC(url, cookie_path=str(cookie)).ping()
            break
        except NodeRPCError:
            if time.time() > deadline:
                process.kill()
                pytest.fail("regtest bitcoind did not start")
            time.sleep(0.5)

    rpc = BitcoinRPC(url, cookie_path=str(cookie))
    rpc.call("createwallet", ["test"])
    wallet = BitcoinRPC(url + "/wallet/test", cookie_path=str(cookie))
    # 200 blocks: many mature coinbase coins, so payments don't chain on one coin
    wallet.call("generatetoaddress", [200, wallet.call("getnewaddress")])
    yield {"url": url, "cookie": str(cookie), "zmq": f"tcp://127.0.0.1:{zmq_port}",
           "wallet": wallet}
    try:
        rpc.call("stop")
        process.wait(30)
    except Exception:
        process.kill()


@pytest.fixture
def feed(node, monkeypatch):
    from src.core.backend.live import monitor

    monkeypatch.setenv("BATMAN_RPC_URL", node["url"])
    monkeypatch.setenv("BATMAN_RPC_COOKIE", node["cookie"])
    monkeypatch.setenv("BATMAN_ZMQ_ENDPOINT", node["zmq"])
    fetch = monitor.build_zmq_fetch()
    wallet = node["wallet"]
    # ZMQ needs a moment to establish the subscription: warm up first
    for _ in range(100):
        wallet.call("sendtoaddress", [wallet.call("getnewaddress"), 0.01])
        time.sleep(0.1)
        if fetch.stats()["received"]:
            break
    time.sleep(0.5)
    fetch(None)
    yield fetch
    fetch.listener.stop()


def send(wallet, n):
    return [wallet.call("sendtoaddress", [wallet.call("getnewaddress"), 0.01]) for _ in range(n)]


def wait_for(fetch, received, timeout=15.0):
    deadline = time.time() + timeout
    while fetch.stats()["received"] < received and time.time() < deadline:
        time.sleep(0.1)


def test_real_node_feed_end_to_end(node, feed, tmp_path):
    from src.core.backend.live import monitor
    from src.core.backend.live.live_store import LiveStore

    wallet = node["wallet"]
    store = LiveStore(tmp_path / "live.sqlite")
    start = feed.stats()

    # Sent while no cycle runs: all processed, with resolved senders
    sent = send(wallet, 30)
    wait_for(feed, start["received"] + 30)
    result = monitor.run_cycle(store, max_tx=None, watchlist=set(), fetch_fn=feed)
    assert result.error is None and result.fetched == 30 and result.valid == 30
    assert set(sent) <= set(store.transactions(limit=10_000)["txid"])

    # Mined before the cycle: found in the latest block; re-announcements skipped
    late = send(wallet, 5)
    wallet.call("generatetoaddress", [1, wallet.call("getnewaddress")])
    time.sleep(2)
    result = monitor.run_cycle(store, max_tx=None, watchlist=set(), fetch_fn=feed)
    assert result.fetched == 5
    assert set(late) <= set(store.transactions(limit=10_000)["txid"])
    stats = feed.stats()
    assert stats["missed"] == 0 and stats["unresolved"] == 0
    assert stats["duplicates"] >= 35  # the mined block re-announced the 35 above


def test_real_node_child_of_unconfirmed_parent(node, feed):
    wallet = node["wallet"]
    sender = wallet.call("getnewaddress")
    parent = wallet.call("sendtoaddress", [sender, 1.0])
    n = [v["n"] for v in wallet.call("getrawtransaction", [parent, 1])["vout"]
         if abs(v["value"] - 1.0) < 1e-9][0]
    psbt = wallet.call("walletcreatefundedpsbt", [[{"txid": parent, "vout": n}],
                                                  [{wallet.call("getnewaddress"): 0.5}], 0,
                                                  {"add_inputs": False}])["psbt"]
    signed = wallet.call("walletprocesspsbt", [psbt])["psbt"]
    child = wallet.call("sendrawtransaction", [wallet.call("finalizepsbt", [signed])["hex"]])
    time.sleep(2)
    frame = feed(None).set_index("txid")
    assert frame.loc[child, "input_addresses"] == sender  # from the mempool parent
    assert frame.loc[child, "fee"] > 0
