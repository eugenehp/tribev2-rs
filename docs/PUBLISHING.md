# Publishing to crates.io

Use **[`scripts/publish.sh`](../scripts/publish.sh)** — modelled after [`../rlx/rig.sh`](../rlx/rig.sh) (header docs, subcommands, preflight).

## Crates (in order)

| # | Crate | Notes |
|---|--------|--------|
| 1 | `tribev2-audio` | RLX Wav2Vec-BERT features |
| 2 | `tribev2-video` | RLX V-JEPA2 features |
| 3 | `tribev2` | Core encoder + `tribev2-infer` / `tribev2-download` bins |

## One-time setup

```bash
cargo login
# Add LICENSE (Apache-2.0) at repo root if missing
```

Ensure root `Cargo.toml` workspace deps include **version** for path crates:

```toml
tribev2-audio = { version = "0.1.0", path = "crates/tribev2-audio", ... }
tribev2-video = { version = "0.1.0", path = "crates/tribev2-video", ... }
```

## Commands

```bash
./scripts/publish.sh doctor      # git clean, login, LICENSE, dry-run all crates
./scripts/publish.sh dry-run     # same checks + package verify, no upload
./scripts/publish.sh publish     # upload (prompts unless -y)
```

Flags: `--allow-dirty`, `--only tribev2-audio`, `-y`.

## After release

Bump **all three** `version =` in crate `Cargo.toml` files and workspace `tribev2-audio` / `tribev2-video` version lines, then publish again in the same order.
