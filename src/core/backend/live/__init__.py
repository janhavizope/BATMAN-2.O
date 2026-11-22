"""
Live monitoring: pull current Bitcoin transactions, map them to the BATMAN
schema, score them with the unsupervised pipeline, and keep a rolling store.

This is the online counterpart to the batch Elliptic++ pipeline. It adds no
new third-party dependency — the feed uses the standard library (urllib) and
the store uses sqlite3.
"""
