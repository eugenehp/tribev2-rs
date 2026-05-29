//! Paths to bundled test / benchmark data.

use std::path::{Path, PathBuf};

/// Root data directory: `TRIBEV2_DATA_DIR` or `<workspace>/data`.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TRIBEV2_DATA_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

pub fn parity_refs_dir() -> PathBuf {
    data_dir().join("parity_refs")
}

pub fn config_path() -> PathBuf {
    data_dir().join("config.yaml")
}

pub fn weights_path() -> PathBuf {
    data_dir().join("model.safetensors")
}

pub fn build_args_path() -> PathBuf {
    data_dir().join("build_args.json")
}

pub fn weights_available() -> bool {
    weights_path().is_file()
}

pub fn parity_refs_available() -> bool {
    parity_refs_dir().join("input_text.bin").is_file()
}

pub fn path_or(p: &Path) -> Option<String> {
    p.to_str().map(|s| s.to_string())
}
