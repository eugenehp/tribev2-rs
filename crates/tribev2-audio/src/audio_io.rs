//! Audio file I/O — load and resample WAV/MP3/FLAC to 16 kHz mono f32.
//!
//! Uses `hound` for WAV reading and `rubato` for sample-rate conversion.

use anyhow::{Context, Result};

/// Load an audio file as mono f32 samples at the target sample rate.
///
/// Supports WAV files directly via `hound`. For MP3/FLAC/OGG, shells out
/// to `ffmpeg` to convert to 16-bit WAV first.
pub fn load_audio(path: &str, target_sr: u32) -> Result<Vec<f32>> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "wav" => load_wav(path, target_sr),
        _ => {
            // Convert to WAV via ffmpeg, then load
            let tmp = tempfile::NamedTempFile::with_suffix(".wav")?;
            let tmp_path = tmp.path().to_str().unwrap();
            let status = std::process::Command::new("ffmpeg")
                .args(["-y", "-i", path])
                .args(["-ar", &target_sr.to_string()])
                .args(["-ac", "1", "-f", "wav"])
                .arg(tmp_path)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .with_context(|| "ffmpeg not found")?;
            if !status.success() {
                anyhow::bail!("ffmpeg failed converting {} to WAV", path);
            }
            load_wav(tmp_path, target_sr)
        }
    }
}

/// Load a WAV file, convert to mono f32, resample to target_sr if needed.
fn load_wav(path: &str, target_sr: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV: {}", path))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let source_sr = spec.sample_rate;

    // Read samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .map(|s| s.unwrap_or(0.0))
                .collect()
        }
    };

    // Convert to mono by averaging channels
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample if needed
    if source_sr == target_sr {
        return Ok(mono);
    }

    resample(&mono, source_sr, target_sr)
}

/// Resample audio using rubato (sinc interpolation).
fn resample(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, Resampler, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_sr as f64 / from_sr as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        input.len(),
        1,
    )?;

    let input_buf = vec![input.to_vec()];
    let output_buf = resampler.process(&input_buf, None)?;

    Ok(output_buf.into_iter().next().unwrap_or_default())
}

/// Get audio duration in seconds.
pub fn audio_duration(path: &str) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .with_context(|| "ffprobe not found")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let dur: f64 = stdout.trim().parse()
        .with_context(|| format!("failed to parse duration: '{}'", stdout.trim()))?;
    Ok(dur)
}
