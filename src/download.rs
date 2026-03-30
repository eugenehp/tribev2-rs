//! HuggingFace Hub weight download for TRIBE v2.
//!
//! Downloads all files needed to run inference:
//! - `config.yaml`       — model architecture config
//! - `model.safetensors` — weights (preferred; tried first)
//! - `best.ckpt`         — PyTorch Lightning checkpoint (fallback)
//! - `build_args.json`   — feature dims / output shape metadata
//!
//! Requires the `hf-download` feature flag.
//!
//! # Example
//! ```rust,ignore
//! use tribev2::download::{DownloadConfig, download_model};
//!
//! let cfg = DownloadConfig {
//!     repo: "eugenehp/tribev2".into(),
//!     output_dir: "./weights".into(),
//!     token: None,       // or Some("hf_...".into()) for private repos
//!     overwrite: false,
//! };
//! let files = download_model(&cfg)?;
//! println!("weights at: {}", files.weights.display());
//! ```

use std::path::{Path, PathBuf};
use anyhow::{Context, Result, bail};
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo, ApiError};

// ── Public types ─────────────────────────────────────────────────────────

/// Configuration for a model download.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// HuggingFace repo ID, e.g. `"eugenehp/tribev2"`.
    pub repo: String,
    /// Local directory to copy files into.
    pub output_dir: PathBuf,
    /// Optional HF token (needed for private / gated repos).
    pub token: Option<String>,
    /// If `false`, skip files that already exist in `output_dir`.
    pub overwrite: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            repo: "eugenehp/tribev2".into(),
            output_dir: PathBuf::from("./weights"),
            token: None,
            overwrite: false,
        }
    }
}

/// Paths of all files that were downloaded (or already present).
#[derive(Debug, Clone)]
pub struct ModelFiles {
    /// `config.yaml`
    pub config: PathBuf,
    /// `model.safetensors` if available, otherwise `best.ckpt`.
    pub weights: PathBuf,
    /// `true` when `weights` points to a native safetensors file.
    pub weights_is_safetensors: bool,
    /// `build_args.json` if it was present in the repo.
    pub build_args: Option<PathBuf>,
}

impl ModelFiles {
    /// Print a human-readable summary to stdout.
    pub fn print_summary(&self) {
        println!("\n── Downloaded files ─────────────────────────────");
        println!("  config     : {}", self.config.display());
        println!("  weights    : {}", self.weights.display());
        if !self.weights_is_safetensors {
            println!("  ⚠  weights are a PyTorch .ckpt — convert with:");
            println!("     python3 -c \"");
            println!("     import torch, safetensors.torch");
            println!("     ckpt = torch.load('{}', map_location='cpu', weights_only=True)",
                self.weights.display());
            println!("     sd = {{k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}}");
            println!("     safetensors.torch.save_file(sd, '{}')",
                self.weights.parent().unwrap_or(Path::new(".")).join("model.safetensors").display());
            println!("     \"");
        }
        if let Some(ref ba) = self.build_args {
            println!("  build_args : {}", ba.display());
        }
        println!("─────────────────────────────────────────────────");
    }
}

// ── Main entry point ──────────────────────────────────────────────────────

/// Download all TRIBE v2 model files to `cfg.output_dir`.
///
/// File resolution order for weights:
/// 1. `model.safetensors`  (ready to use with `WeightMap::from_safetensors`)
/// 2. `best.ckpt`          (needs Python conversion; a warning is printed)
///
/// Files that already exist are skipped unless `cfg.overwrite` is `true`.
pub fn download_model(cfg: &DownloadConfig) -> Result<ModelFiles> {
    std::fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("creating output dir {:?}", cfg.output_dir))?;

    let api = build_api(cfg)?;
    let repo = api.model(cfg.repo.clone());

    // ── config.yaml ───────────────────────────────────────────────────
    let config = fetch_file(&repo, "config.yaml", &cfg.output_dir, cfg.overwrite)
        .context("downloading config.yaml")?;

    // ── weights: safetensors preferred, ckpt fallback ─────────────────
    let (weights, weights_is_safetensors) =
        fetch_weights(&repo, &cfg.output_dir, cfg.overwrite)
            .context("downloading model weights")?;

    // ── build_args.json (optional) ────────────────────────────────────
    let build_args = fetch_optional_file(&repo, "build_args.json", &cfg.output_dir, cfg.overwrite)
        .context("checking for build_args.json")?;

    Ok(ModelFiles { config, weights, weights_is_safetensors, build_args })
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn build_api(cfg: &DownloadConfig) -> Result<Api> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(ref token) = cfg.token {
        builder = builder.with_token(Some(token.clone()));
    }
    builder.build().context("building HuggingFace API client")
}

/// Download a single file; return its local path.
fn fetch_file(
    repo: &ApiRepo,
    filename: &str,
    out_dir: &Path,
    overwrite: bool,
) -> Result<PathBuf> {
    let dest = out_dir.join(filename);
    if dest.exists() && !overwrite {
        println!("  ✓ {filename} (already present, skipping)");
        return Ok(dest);
    }
    println!("  ↓ {filename}");
    let cached = repo.get(filename)
        .with_context(|| format!("fetching {filename} from HF Hub"))?;
    std::fs::copy(&cached, &dest)
        .with_context(|| format!("copying {filename} to {}", dest.display()))?;
    println!("    → {}", dest.display());
    Ok(dest)
}

/// Attempt to download `filename`; return `None` silently if it doesn't exist.
fn fetch_optional_file(
    repo: &ApiRepo,
    filename: &str,
    out_dir: &Path,
    overwrite: bool,
) -> Result<Option<PathBuf>> {
    let dest = out_dir.join(filename);
    if dest.exists() && !overwrite {
        println!("  ✓ {filename} (already present, skipping)");
        return Ok(Some(dest));
    }
    match repo.get(filename) {
        Ok(cached) => {
            println!("  ↓ {filename}");
            std::fs::copy(&cached, &dest)
                .with_context(|| format!("copying {filename} to {}", dest.display()))?;
            println!("    → {}", dest.display());
            Ok(Some(dest))
        }
        // 404 / not found — silently skip
        Err(e) if is_not_found(&e) => {
            println!("  – {filename} not found in repo (optional, skipping)");
            Ok(None)
        }
        Err(e) => Err(e).with_context(|| format!("fetching optional file {filename}")),
    }
}

/// Try `model.safetensors`, then `best.ckpt`.
fn fetch_weights(
    repo: &ApiRepo,
    out_dir: &Path,
    overwrite: bool,
) -> Result<(PathBuf, bool)> {
    // --- safetensors ---
    let st_dest = out_dir.join("model.safetensors");
    if st_dest.exists() && !overwrite {
        println!("  ✓ model.safetensors (already present, skipping)");
        return Ok((st_dest, true));
    }
    match repo.get("model.safetensors") {
        Ok(cached) => {
            println!("  ↓ model.safetensors");
            std::fs::copy(&cached, &st_dest)
                .context("copying model.safetensors")?;
            println!("    → {}", st_dest.display());
            return Ok((st_dest, true));
        }
        Err(e) if is_not_found(&e) => {
            println!("  – model.safetensors not found, trying best.ckpt …");
        }
        Err(e) => {
            return Err(e).context("fetching model.safetensors");
        }
    }

    // --- ckpt fallback ---
    let ckpt_dest = out_dir.join("best.ckpt");
    if ckpt_dest.exists() && !overwrite {
        println!("  ✓ best.ckpt (already present, skipping)");
        return Ok((ckpt_dest, false));
    }
    match repo.get("best.ckpt") {
        Ok(cached) => {
            println!("  ↓ best.ckpt");
            std::fs::copy(&cached, &ckpt_dest)
                .context("copying best.ckpt")?;
            println!("    → {}", ckpt_dest.display());
            Ok((ckpt_dest, false))
        }
        Err(e) if is_not_found(&e) => {
            bail!("neither model.safetensors nor best.ckpt found in repo '{}'", "?");
        }
        Err(e) => Err(e).context("fetching best.ckpt"),
    }
}

/// Heuristic: treat an error as "file not found" when its message contains a
/// 404 status or "not found" text (hf-hub surfaces 404s inside RequestError).
fn is_not_found(e: &ApiError) -> bool {
    let msg = e.to_string().to_lowercase();
    msg.contains("404") || msg.contains("not found") || msg.contains("entry not found")
}
