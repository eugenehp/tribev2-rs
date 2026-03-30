//! Download TRIBE v2 weights from HuggingFace Hub.
//!
//! Usage:
//!   tribev2-download [OPTIONS]
//!
//! Examples:
//!   # Download public model weights
//!   tribev2-download --repo facebook/tribev2 --output ./weights
//!
//!   # Download from a private / gated repo
//!   tribev2-download --repo my-org/tribev2-private --token hf_xxxx --output ./weights
//!
//!   # Force re-download even if files already exist
//!   tribev2-download --overwrite

use clap::Parser;
use std::path::PathBuf;
use tribev2::download::{DownloadConfig, download_model};

#[derive(Parser, Debug)]
#[command(
    name = "tribev2-download",
    about = "Download TRIBE v2 model weights from HuggingFace Hub",
    long_about = "\
Downloads all files needed to run tribev2-infer:
  • config.yaml        — model architecture config
  • model.safetensors  — weights (preferred; used directly by the inference engine)
  • best.ckpt          — PyTorch Lightning checkpoint (fallback if safetensors absent)
  • build_args.json    — feature-dimension / output-shape metadata (optional)

If model.safetensors is not present in the repo the tool falls back to best.ckpt
and prints the Python one-liner needed to convert it to safetensors format."
)]
struct Args {
    /// HuggingFace repo ID (e.g. \"facebook/tribev2\")
    #[arg(long, default_value = "facebook/tribev2")]
    repo: String,

    /// Local directory to save files into
    #[arg(long, short, default_value = "./weights")]
    output: PathBuf,

    /// HuggingFace API token (required for private / gated repos).
    /// Can also be set via the HF_TOKEN environment variable.
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,

    /// Re-download and overwrite files that already exist locally
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let cfg = DownloadConfig {
        repo: args.repo.clone(),
        output_dir: args.output.clone(),
        token: args.token,
        overwrite: args.overwrite,
    };

    println!("Downloading TRIBE v2 model from HuggingFace");
    println!("  repo   : {}", cfg.repo);
    println!("  output : {}", cfg.output_dir.display());
    if cfg.overwrite {
        println!("  mode   : overwrite");
    }
    println!();

    let files = download_model(&cfg)?;
    files.print_summary();

    Ok(())
}
