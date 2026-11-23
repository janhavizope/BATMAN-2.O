# Shrinking the repository history (runbook)

**Status: prepared and dry-run tested. Not run.** Running it rewrites every commit and needs a force-push, so the team has to agree on it first and do it as one announced step.

## Why

A fresh `git clone` downloads all of history, not only the current files. History holds large files that are no longer used:

| File(s) in history | Size | Still on `main`? |
|---|---|---|
| `data/elliptic_plus/*.parquet`, old versions from before Git LFS | ~52 MB | Current versions yes (already in LFS) |
| `data/dev/dev_dataset_50k.csv` and `src/core/data/dev/dev_dataset_50k.csv` (synthetic, 3 copies) | ~62 MB | No |
| `illicit_tx_rf_model.joblib` (old synthetic-data model, 2 copies) | ~39 MB | No |
| `synthetic_50k.csv`, `synthetic_bitcoin_transactions.csv` (2 copies) | ~19 MB | No |

## Dry-run results

Measured on a throwaway clone with every branch rewritten (November 2026):

| Option | History downloaded on clone | Added Git LFS storage |
|---|---|---|
| Today | **107 MB** | — |
| A: move the old parquet versions into Git LFS (the original TODO) | 55 MB | +52 MB |
| **B: A, plus remove the deleted datasets and models from history** | **4.8 MB** | +52 MB |

In both options the files on `main` are identical, byte for byte. In option B, 3 commits that only added or removed the deleted files disappear (`main` goes from 42 to 39 commits). **Option B is recommended.**

## What it changes for everyone

- **Every commit ID changes.** Links to old commits (in PR descriptions, chat, docs) stop working.
- **Everyone must re-clone.** Running `git pull` on an old clone would mix the old and new histories. Delete old clones after the switch.
- **Merge or close open pull requests first.** Their branches are rewritten too, so any PR left open would show the whole history as changed.
- **GitHub keeps the old history in pull-request refs** (`refs/pull/*`, which can't be deleted). Fresh clones don't download those, so they get the small history. The repository size GitHub reports only drops if GitHub Support runs garbage collection, which you can request but don't have to.
- **The old parquet versions are uploaded to Git LFS storage** (+52 MB). That counts against the LFS storage quota. Only someone checking out an old commit downloads them.

## Steps

Run these on one machine, with Git LFS and [git-filter-repo](https://github.com/newren/git-filter-repo) installed (`pip install git-filter-repo`).

1. **Announce a freeze.** Everyone pushes their work and stops pushing. Merge or close every open PR.
2. **Make a fresh full copy, with every branch local, and keep a backup:**
   ```bash
   git clone https://github.com/janhavizope/BATMAN-2.O.git batman-rewrite
   cd batman-rewrite
   git lfs fetch --all
   for b in $(git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin | grep -v '^HEAD$'); do
     [ "$b" = main ] || git branch "$b" "origin/$b"
   done
   git bundle create ../batman-backup.bundle --all     # full backup; keep it
   ```
3. **Option A: move the old parquet versions into Git LFS:**
   ```bash
   git lfs migrate import --everything --include="data/elliptic_plus/*.parquet" --yes
   ```
4. **Option B: remove the deleted datasets and models from all history:**
   ```bash
   git filter-repo --force --invert-paths \
     --path data/dev/ --path bitcoin-monitor/ --path src/core/data/dev/ \
     --path src/core/backend/ml/illicit_tx_rf_model.joblib \
     --path synthetic_50k.csv --path src/core/backend/ml/synthetic_bitcoin_transactions.csv
   ```
   git-filter-repo removes the `origin` remote on purpose. Add it back:
   `git remote add origin https://github.com/janhavizope/BATMAN-2.O.git`
5. **Check before pushing:**
   ```bash
   git gc --prune=now --aggressive
   git count-objects -vH                         # size-pack should be about 5 MB
   git clone -q ../batman-backup.bundle ../batman-before
   git -C ../batman-before checkout -q main
   diff -r --exclude=.git ../batman-before .     # expect no output: main's files unchanged
   ```
   Also run the test suite (`pytest`) in the rewritten clone.
6. **Upload the LFS objects first, then force-push every branch:**
   ```bash
   git lfs push --all origin
   git push --force --all origin
   git push --force --tags origin
   ```
   Branch protection on `main` (if enabled) must allow this push for the duration.
7. **Everyone re-clones** (`git lfs install && git clone ...`) and deletes their old clone.

**To undo:**
- **Before step 6,** delete the `batman-rewrite` folder.
- **After step 6,** push the backup back:
  ```bash
  git clone --mirror ../batman-backup.bundle restore.git
  cd restore.git
  git push --force --all https://github.com/janhavizope/BATMAN-2.O.git
  ```
