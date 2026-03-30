//! Event types and transforms for TRIBE v2 inference pipeline.
//!
//! Mirrors the Python `eventstransforms.py` and `demo_utils.py` pipeline:
//! - Parse text/audio/video input files
//! - Extract word-level timing from audio (via whisper CLI)
//! - Build events DataFrames for feature extraction
//!
//! The pipeline converts raw media into timed word events:
//! 1. Audio → whisper transcription → word timestamps
//! 2. Video → extract audio → whisper → word timestamps
//! 3. Text → TTS → audio → whisper → word timestamps
//!
//! For text-only inference without audio alignment, words are assigned
//! uniform timing across the duration.

use std::path::Path;
use anyhow::{Context, Result};

/// An event in the timeline.
#[derive(Debug, Clone)]
pub struct Event {
    /// Event type: "Word", "Audio", "Video", "Text", "Fmri"
    pub event_type: String,
    /// Start time in seconds.
    pub start: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Text content (for Word events).
    pub text: Option<String>,
    /// File path (for Audio/Video events).
    pub filepath: Option<String>,
    /// Timeline identifier.
    pub timeline: String,
    /// Subject identifier.
    pub subject: String,
    /// Sentence context (for Word events).
    pub sentence: Option<String>,
    /// Context window (for Word events with surrounding context).
    pub context: Option<String>,
}

impl Event {
    pub fn word(text: &str, start: f64, duration: f64) -> Self {
        Self {
            event_type: "Word".into(),
            start,
            duration,
            text: Some(text.into()),
            filepath: None,
            timeline: "default".into(),
            subject: "default".into(),
            sentence: None,
            context: None,
        }
    }

    pub fn audio(filepath: &str, start: f64, duration: f64) -> Self {
        Self {
            event_type: "Audio".into(),
            start,
            duration,
            text: None,
            filepath: Some(filepath.into()),
            timeline: "default".into(),
            subject: "default".into(),
            sentence: None,
            context: None,
        }
    }

    pub fn video(filepath: &str, start: f64, duration: f64) -> Self {
        Self {
            event_type: "Video".into(),
            start,
            duration,
            text: None,
            filepath: Some(filepath.into()),
            timeline: "default".into(),
            subject: "default".into(),
            sentence: None,
            context: None,
        }
    }
}

/// A collection of events forming a timeline.
#[derive(Debug, Clone, Default)]
pub struct EventList {
    pub events: Vec<Event>,
}

impl EventList {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Get all word events, sorted by start time.
    pub fn words(&self) -> Vec<&Event> {
        let mut words: Vec<&Event> = self.events.iter()
            .filter(|e| e.event_type == "Word")
            .collect();
        words.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
        words
    }

    /// Get total timeline duration.
    pub fn duration(&self) -> f64 {
        self.events.iter()
            .map(|e| e.start + e.duration)
            .fold(0.0, f64::max)
    }

    /// Convert word events to (text, start_time) pairs for feature extraction.
    pub fn words_timed(&self) -> Vec<(String, f64)> {
        self.words().iter()
            .filter_map(|e| {
                e.text.as_ref().map(|t| (t.clone(), e.start))
            })
            .collect()
    }

    /// Add sentence context to word events.
    ///
    /// Groups consecutive words into sentences (split by punctuation),
    /// then assigns the sentence text to each word's `sentence` field.
    pub fn add_sentence_context(&mut self) {
        let words: Vec<usize> = self.events.iter()
            .enumerate()
            .filter(|(_, e)| e.event_type == "Word")
            .map(|(i, _)| i)
            .collect();

        if words.is_empty() { return; }

        // Simple sentence splitting: group by punctuation
        let mut sentence_start = 0;
        let mut sentence_words: Vec<(usize, usize)> = Vec::new(); // (start_idx, end_idx) in words vec

        for (wi, &ei) in words.iter().enumerate() {
            let text = self.events[ei].text.as_deref().unwrap_or("");
            let ends_sentence = text.ends_with('.') || text.ends_with('!') ||
                text.ends_with('?') || text.ends_with(';');

            if ends_sentence || wi == words.len() - 1 {
                sentence_words.push((sentence_start, wi + 1));
                sentence_start = wi + 1;
            }
        }

        // Assign sentence text
        for (start, end) in sentence_words {
            let sentence: String = (start..end)
                .map(|wi| self.events[words[wi]].text.as_deref().unwrap_or(""))
                .collect::<Vec<_>>()
                .join(" ");

            for wi in start..end {
                self.events[words[wi]].sentence = Some(sentence.clone());
            }
        }
    }

    /// Add surrounding context window to word events.
    ///
    /// Each word gets a `context` field with up to `max_context_len` characters
    /// of surrounding words.
    pub fn add_context(&mut self, max_context_len: usize) {
        let word_indices: Vec<usize> = self.events.iter()
            .enumerate()
            .filter(|(_, e)| e.event_type == "Word")
            .map(|(i, _)| i)
            .collect();

        let all_words: Vec<String> = word_indices.iter()
            .map(|&i| self.events[i].text.as_deref().unwrap_or("").to_string())
            .collect();

        for (pos, &ei) in word_indices.iter().enumerate() {
            let mut context = String::new();
            // Add preceding words
            let mut start = pos;
            while start > 0 {
                start -= 1;
                let candidate = format!("{} {}", all_words[start], context.trim_start());
                if candidate.len() > max_context_len {
                    break;
                }
                context = candidate;
            }
            // Add current and following words
            let current = &all_words[pos];
            if context.is_empty() {
                context = current.clone();
            } else {
                context = format!("{} {}", context.trim(), current);
            }
            let mut end = pos + 1;
            while end < all_words.len() {
                let candidate = format!("{} {}", context, all_words[end]);
                if candidate.len() > max_context_len {
                    break;
                }
                context = candidate;
                end += 1;
            }

            self.events[ei].context = Some(context);
        }
    }
}

/// Create word events from plain text with uniform timing.
///
/// Words are evenly spaced across the given duration.
pub fn text_to_events(text: &str, total_duration: f64) -> EventList {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n == 0 {
        return EventList::new();
    }

    let word_duration = total_duration / n as f64;
    let mut events = EventList::new();
    for (i, word) in words.iter().enumerate() {
        events.push(Event::word(word, i as f64 * word_duration, word_duration));
    }
    events
}

/// Parse a whisper JSON transcript into word events.
///
/// Expected format (whisperX output):
/// ```json
/// { "segments": [ { "text": "...", "words": [ {"word": "...", "start": 0.0, "end": 0.5}, ... ] } ] }
/// ```
pub fn parse_whisper_json(json_str: &str, time_offset: f64) -> Result<EventList> {
    let v: serde_json::Value = serde_json::from_str(json_str)
        .with_context(|| "failed to parse whisper JSON")?;

    let mut events = EventList::new();

    let segments = v.get("segments")
        .and_then(|s| s.as_array())
        .unwrap_or(&Vec::new())
        .clone();

    for segment in &segments {
        let sentence = segment.get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .trim()
            .replace('"', "");

        let words = segment.get("words")
            .and_then(|w| w.as_array())
            .cloned()
            .unwrap_or_default();

        for word_obj in &words {
            let text = word_obj.get("word")
                .and_then(|w| w.as_str())
                .unwrap_or("")
                .trim()
                .replace('"', "");

            let start = word_obj.get("start")
                .and_then(|s| s.as_f64())
                .unwrap_or(0.0);

            let end = word_obj.get("end")
                .and_then(|e| e.as_f64())
                .unwrap_or(start);

            if text.is_empty() || !word_obj.get("start").is_some() {
                continue;
            }

            let mut event = Event::word(&text, start + time_offset, end - start);
            event.sentence = Some(sentence.clone());
            events.push(event);
        }
    }

    Ok(events)
}

/// Run whisper transcription on an audio file.
///
/// Shells out to `whisperx` CLI (must be installed via `pip install whisperx`).
/// Falls back to `whisper` if `whisperx` is not available.
///
/// Returns word-level events.
pub fn transcribe_audio(
    audio_path: &str,
    language: &str,
    time_offset: f64,
) -> Result<EventList> {
    let audio_path = Path::new(audio_path);
    if !audio_path.exists() {
        anyhow::bail!("audio file not found: {}", audio_path.display());
    }

    // Check for cached transcript (.json next to audio file)
    let json_path = audio_path.with_extension("json");
    if json_path.exists() {
        let json_str = std::fs::read_to_string(&json_path)?;
        return parse_whisper_json(&json_str, time_offset);
    }

    // Try whisperx first, then whisper
    let temp_dir = tempfile::tempdir()
        .with_context(|| "failed to create temp dir for whisper output")?;

    let lang_code = match language {
        "english" => "en",
        "french" => "fr",
        "spanish" => "es",
        "dutch" => "nl",
        "chinese" => "zh",
        other => other,
    };

    let output = std::process::Command::new("whisperx")
        .arg(audio_path.to_str().unwrap_or(""))
        .args(["--model", "large-v3"])
        .args(["--language", lang_code])
        .args(["--output_dir", temp_dir.path().to_str().unwrap_or("")])
        .args(["--output_format", "json"])
        .output();

    let json_content = match output {
        Ok(out) if out.status.success() => {
            let stem = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
            let result_path = temp_dir.path().join(format!("{}.json", stem));
            std::fs::read_to_string(&result_path)
                .with_context(|| format!("whisperx ran but output not found: {}", result_path.display()))?
        }
        _ => {
            // Try plain whisper as fallback
            let output = std::process::Command::new("whisper")
                .arg(audio_path.to_str().unwrap_or(""))
                .args(["--model", "large-v3"])
                .args(["--language", lang_code])
                .args(["--output_dir", temp_dir.path().to_str().unwrap_or("")])
                .args(["--output_format", "json"])
                .args(["--word_timestamps", "True"])
                .output()
                .with_context(|| "neither whisperx nor whisper found. Install: pip install whisperx")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!("whisper transcription failed: {}", stderr);
            }

            let stem = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
            let result_path = temp_dir.path().join(format!("{}.json", stem));
            std::fs::read_to_string(&result_path)
                .with_context(|| "whisper ran but output not found")?
        }
    };

    // Cache the result
    if let Err(e) = std::fs::write(&json_path, &json_content) {
        eprintln!("Warning: could not cache transcript: {}", e);
    }

    parse_whisper_json(&json_content, time_offset)
}

/// Extract audio from a video file using ffmpeg.
///
/// Returns path to the extracted WAV file.
pub fn extract_audio_from_video(video_path: &str, output_dir: &str) -> Result<String> {
    let video = Path::new(video_path);
    if !video.exists() {
        anyhow::bail!("video file not found: {}", video.display());
    }

    std::fs::create_dir_all(output_dir)?;
    let stem = video.file_stem().and_then(|s| s.to_str()).unwrap_or("video");
    let wav_path = format!("{}/{}.wav", output_dir, stem);

    if Path::new(&wav_path).exists() {
        return Ok(wav_path);
    }

    let output = std::process::Command::new("ffmpeg")
        .args(["-i", video_path])
        .args(["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])
        .arg(&wav_path)
        .args(["-y"])  // overwrite
        .output()
        .with_context(|| "ffmpeg not found. Install: brew install ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed: {}", stderr);
    }

    Ok(wav_path)
}

/// Get audio duration in seconds using ffprobe.
pub fn get_audio_duration(path: &str) -> Result<f64> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "quiet"])
        .args(["-show_entries", "format=duration"])
        .args(["-of", "csv=p=0"])
        .arg(path)
        .output()
        .with_context(|| "ffprobe not found")?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let duration: f64 = stdout.trim().parse()
        .with_context(|| format!("failed to parse duration from ffprobe: '{}'", stdout.trim()))?;
    Ok(duration)
}

/// Get video duration in seconds.
pub fn get_video_duration(path: &str) -> Result<f64> {
    get_audio_duration(path)
}

/// Build a complete events pipeline from a media file.
///
/// Mirrors Python `TribeModel.get_events_dataframe()`:
/// - Audio → transcribe → word events
/// - Video → extract audio → transcribe → word events
/// - Text → uniform word timing
///
/// Returns events with sentence and context annotations.
pub fn build_events_from_media(
    text_path: Option<&str>,
    audio_path: Option<&str>,
    video_path: Option<&str>,
    cache_dir: &str,
    language: &str,
    max_context_len: usize,
) -> Result<EventList> {
    std::fs::create_dir_all(cache_dir)?;

    let mut events = if let Some(text_path) = text_path {
        let text = std::fs::read_to_string(text_path)
            .with_context(|| format!("failed to read text: {}", text_path))?;
        // Estimate duration: ~3 words/second for speech
        let n_words = text.split_whitespace().count();
        let duration = n_words as f64 / 3.0;
        text_to_events(&text, duration)

    } else if let Some(audio_path) = audio_path {
        let duration = get_audio_duration(audio_path)?;
        let mut events = transcribe_audio(audio_path, language, 0.0)?;
        events.push(Event::audio(audio_path, 0.0, duration));
        events

    } else if let Some(video_path) = video_path {
        let duration = get_video_duration(video_path)?;
        let wav_path = extract_audio_from_video(video_path, cache_dir)?;
        let mut events = transcribe_audio(&wav_path, language, 0.0)?;
        events.push(Event::video(video_path, 0.0, duration));
        events.push(Event::audio(&wav_path, 0.0, duration));
        events

    } else {
        anyhow::bail!("one of text_path, audio_path, or video_path must be provided");
    };

    // Add sentence and context annotations
    events.add_sentence_context();
    events.add_context(max_context_len);

    Ok(events)
}

/// Create a static-image video using ffmpeg (mirrors `CreateVideosFromImages`).
///
/// Takes an image file and duration, produces an MP4 video of that image
/// held for the given duration. Used to feed static images to V-JEPA.
pub fn create_video_from_image(
    image_path: &str,
    duration: f64,
    fps: u32,
    output_dir: &str,
) -> Result<String> {
    let image = Path::new(image_path);
    if !image.exists() {
        anyhow::bail!("image file not found: {}", image.display());
    }
    std::fs::create_dir_all(output_dir)?;
    let stem = image.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let mp4_path = format!("{}/{}.mp4", output_dir, stem);

    if Path::new(&mp4_path).exists() {
        return Ok(mp4_path);
    }

    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loop", "1"])
        .args(["-i", image_path])
        .args(["-c:v", "libx264", "-t", &format!("{:.2}", duration)])
        .args(["-pix_fmt", "yuv420p", "-r", &fps.to_string()])
        .arg(&mp4_path)
        .status()
        .with_context(|| "ffmpeg not found")?;

    if !status.success() {
        anyhow::bail!("ffmpeg failed creating video from image");
    }
    Ok(mp4_path)
}

/// Remove duplicate events based on specified fields.
///
/// Mirrors `RemoveDuplicates` transform.
pub fn remove_duplicate_events(events: &mut EventList, by_fields: &[&str]) {
    let mut seen = std::collections::HashSet::new();
    events.events.retain(|e| {
        let mut key = String::new();
        for field in by_fields {
            match *field {
                "start" => key.push_str(&format!("{:.6}", e.start)),
                "duration" => key.push_str(&format!("{:.6}", e.duration)),
                "text" => key.push_str(e.text.as_deref().unwrap_or("")),
                "filepath" => key.push_str(e.filepath.as_deref().unwrap_or("")),
                "type" | "event_type" => key.push_str(&e.event_type),
                _ => {}
            }
            key.push('|');
        }
        seen.insert(key)
    });
}

/// Get all words from an event list within a time range.
///
/// Mirrors `plotting/utils.py get_words()`.
pub fn get_words_in_range(
    events: &EventList,
    start: f64,
    end: f64,
    remove_punctuation: bool,
) -> Vec<String> {
    events.events.iter()
        .filter(|e| e.event_type == "Word" && e.start >= start && e.start < end)
        .filter_map(|e| {
            e.text.as_ref().map(|t| {
                if remove_punctuation {
                    t.chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect()
                } else {
                    t.clone()
                }
            })
        })
        .collect()
}

/// Check if an event list contains any Audio events.
///
/// Mirrors `plotting/utils.py has_audio()`.
pub fn has_audio(events: &EventList) -> bool {
    events.events.iter().any(|e| e.event_type == "Audio")
}

/// Check if an event list contains any Video events.
///
/// Mirrors `plotting/utils.py has_video()`.
pub fn has_video(events: &EventList) -> bool {
    events.events.iter().any(|e| e.event_type == "Video")
}

/// Get the first audio file path from events.
pub fn get_audio_path(events: &EventList) -> Option<String> {
    events.events.iter()
        .find(|e| e.event_type == "Audio")
        .and_then(|e| e.filepath.clone())
}

/// Get the first video file path from events.
pub fn get_video_path(events: &EventList) -> Option<String> {
    events.events.iter()
        .find(|e| e.event_type == "Video")
        .and_then(|e| e.filepath.clone())
}

/// Get concatenated text from events within a time range.
///
/// Mirrors `plotting/utils.py get_text()`.
pub fn get_text_in_range(
    events: &EventList,
    start: f64,
    end: f64,
) -> String {
    get_words_in_range(events, start, end, true).join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_events() {
        let events = text_to_events("The quick brown fox", 4.0);
        assert_eq!(events.events.len(), 4);
        assert_eq!(events.events[0].text.as_deref(), Some("The"));
        assert!((events.events[0].start - 0.0).abs() < 1e-6);
        assert!((events.events[1].start - 1.0).abs() < 1e-6);
        assert!((events.events[0].duration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_words_timed() {
        let events = text_to_events("Hello world", 2.0);
        let timed = events.words_timed();
        assert_eq!(timed.len(), 2);
        assert_eq!(timed[0].0, "Hello");
        assert!((timed[0].1 - 0.0).abs() < 1e-6);
        assert!((timed[1].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_sentence_context() {
        let mut events = text_to_events("Hello world. Foo bar.", 4.0);
        events.add_sentence_context();
        // "world." ends a sentence
        let words = events.words();
        assert!(words[0].sentence.is_some());
        assert!(words[1].sentence.as_deref().unwrap().contains("Hello"));
        assert!(words[1].sentence.as_deref().unwrap().contains("world."));
    }

    #[test]
    fn test_add_context() {
        let mut events = text_to_events("A B C D E", 5.0);
        events.add_context(20);
        let words = events.words();
        assert!(words[2].context.is_some());
        let ctx = words[2].context.as_deref().unwrap();
        assert!(ctx.contains("C"));
    }

    #[test]
    fn test_parse_whisper_json() {
        let json = r#"{"segments": [{"text": "Hello world", "words": [{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world", "start": 0.5, "end": 1.0}]}]}"#;
        let events = parse_whisper_json(json, 0.0).unwrap();
        assert_eq!(events.events.len(), 2);
        assert_eq!(events.events[0].text.as_deref(), Some("Hello"));
        assert!((events.events[0].start - 0.0).abs() < 1e-6);
        assert!((events.events[0].duration - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_whisper_json_with_offset() {
        let json = r#"{"segments": [{"text": "Hi", "words": [{"word": "Hi", "start": 1.0, "end": 1.5}]}]}"#;
        let events = parse_whisper_json(json, 10.0).unwrap();
        assert!((events.events[0].start - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_event_duration() {
        let mut events = EventList::new();
        events.push(Event::word("A", 0.0, 1.0));
        events.push(Event::word("B", 2.0, 1.5));
        assert!((events.duration() - 3.5).abs() < 1e-6);
    }
}
