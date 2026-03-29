//! Download TRIBE v2 weights from HuggingFace Hub and convert to safetensors.
//!
//! Usage:
//!   tribev2-download --repo facebook/tribev2 --output ./weights
//!
//! This downloads `config.yaml` and `best.ckpt` from the HuggingFace Hub.
//! The .ckpt file must then be converted to safetensors using:
//!   python3 -c "
//!   import torch, safetensors.torch
//!   ckpt = torch.load('best.ckpt', map_location='cpu', weights_only=True)
//!   sd = {k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}
//!   safetensors.torch.save_file(sd, 'model.safetensors')
//!   "
//!
//! Alternatively, use the Rust convert functionality (planned).

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Download TRIBE v2 model from HuggingFace Hub")]
struct Args {
    /// HuggingFace repo ID (e.g. "facebook/tribev2")
    #[arg(long, default_value = "facebook/tribev2")]
    repo: String,

    /// Output directory
    #[arg(long, default_value = "./weights")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let out_dir = PathBuf::from(&args.output);
    std::fs::create_dir_all(&out_dir)?;

    println!("Downloading from HuggingFace: {}", args.repo);

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?;

    let repo = api.model(args.repo.clone());

    // Download config.yaml
    println!("Downloading config.yaml...");
    let config_path = repo.get("config.yaml")?;
    let dest = out_dir.join("config.yaml");
    std::fs::copy(&config_path, &dest)?;
    println!("  → {}", dest.display());

    // Download best.ckpt
    println!("Downloading best.ckpt...");
    let ckpt_path = repo.get("best.ckpt")?;
    let dest = out_dir.join("best.ckpt");
    std::fs::copy(&ckpt_path, &dest)?;
    println!("  → {}", dest.display());

    println!("\nDownload complete!");
    println!("\nTo convert checkpoint to safetensors, run:");
    println!("  python3 -c \"");
    println!("  import torch, safetensors.torch");
    println!("  ckpt = torch.load('{}', map_location='cpu', weights_only=True)", out_dir.join("best.ckpt").display());
    println!("  sd = {{k.removeprefix('model.'): v for k, v in ckpt['state_dict'].items()}}");
    println!("  safetensors.torch.save_file(sd, '{}')\"", out_dir.join("model.safetensors").display());

    Ok(())
}
