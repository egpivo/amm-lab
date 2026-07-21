//! SHA-256 provenance manifests for new Rust artifacts.
//!
//! Historical freeze manifests are never edited by this module.

use crate::pstt::error::{PsttError, Result};
use crate::pstt::schema::{FileDigest, Manifest};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

const CHUNK: usize = 1 << 20;

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|e| PsttError::io(path, e))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = reader.read(&mut buf).map_err(|e| PsttError::io(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn file_digest(base: &Path, path: &Path) -> Result<FileDigest> {
    let rel = path
        .strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    let meta = fs::metadata(path).map_err(|e| PsttError::io(path, e))?;
    Ok(FileDigest {
        path: rel,
        bytes: meta.len(),
        sha256: sha256_file(path)?,
    })
}

/// Build a manifest over explicit paths, sorted by relative path after fixed entries.
pub fn build_manifest(base: &Path, paths: &[PathBuf]) -> Result<Manifest> {
    let mut files = Vec::new();
    for path in paths {
        if !path.is_file() {
            return Err(PsttError::MissingPath(path.clone()));
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if name.ends_with(".part") || name.ends_with(".missing") {
            continue;
        }
        files.push(file_digest(base, path)?);
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let total_bytes = files.iter().map(|f| f.bytes).sum();
    let file_count = files.len();
    Ok(Manifest {
        files,
        file_count,
        total_bytes,
    })
}

pub fn write_manifest_json(path: &Path, manifest: &Manifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| PsttError::io(parent, e))?;
    }
    let body = serde_json::to_string_pretty(manifest)?;
    // Canonical-ish JSON: serde_json pretty uses 2-space indent; sort already applied.
    let mut file = File::create(path).map_err(|e| PsttError::io(path, e))?;
    file.write_all(body.as_bytes())
        .map_err(|e| PsttError::io(path, e))?;
    file.write_all(b"\n").map_err(|e| PsttError::io(path, e))?;
    Ok(())
}

/// Parse a freeze-manifest line: `<sha256><whitespace><path>`.
pub fn parse_freeze_line(line: &str) -> Result<(String, String)> {
    let line = line.trim();
    if line.is_empty() {
        return Err(PsttError::parse("empty freeze line"));
    }
    let mut parts = line.split_whitespace();
    let digest = parts
        .next()
        .ok_or_else(|| PsttError::parse("missing digest"))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| PsttError::parse("missing path"))?
        .to_string();
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PsttError::parse(format!("invalid sha256 digest: {digest}")));
    }
    if parts.next().is_some() {
        return Err(PsttError::parse("too many freeze-line fields"));
    }
    Ok((digest.to_ascii_lowercase(), path))
}

pub fn verify_freeze_manifest(manifest_path: &Path, root: &Path) -> Result<usize> {
    let text = fs::read_to_string(manifest_path).map_err(|e| PsttError::io(manifest_path, e))?;
    let mut checked = 0usize;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (expected, rel) = parse_freeze_line(line).map_err(|e| {
            PsttError::parse(format!("{}:{}: {e}", manifest_path.display(), lineno + 1))
        })?;
        let path = root.join(&rel);
        let actual = sha256_file(&path)?;
        if actual != expected {
            return Err(PsttError::invariant(format!(
                "hash mismatch for {rel}: expected {expected}, got {actual}"
            )));
        }
        checked += 1;
    }
    Ok(checked)
}

/// Refuse to write into a nonempty directory unless `allow_nonempty` is set.
pub fn require_empty_output_dir(path: &Path, allow_nonempty: bool) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path).map_err(|e| PsttError::io(path, e))?;
        return Ok(());
    }
    if !path.is_dir() {
        return Err(PsttError::schema(format!(
            "output path is not a directory: {}",
            path.display()
        )));
    }
    if allow_nonempty {
        return Ok(());
    }
    let mut rd = fs::read_dir(path).map_err(|e| PsttError::io(path, e))?;
    if rd.next().is_some() {
        return Err(PsttError::NonEmptyOutput(path.to_path_buf()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sha_and_freeze_line_roundtrip() {
        let dir = std::env::temp_dir().join(format!("pstt_manifest_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        {
            let mut f = File::create(&file).unwrap();
            f.write_all(b"hello\n").unwrap();
        }
        let digest = sha256_file(&file).unwrap();
        assert_eq!(digest.len(), 64);
        let (d, p) = parse_freeze_line(&format!("{digest}  a.txt")).unwrap();
        assert_eq!(d, digest);
        assert_eq!(p, "a.txt");
        let man = build_manifest(&dir, &[file]).unwrap();
        assert_eq!(man.file_count, 1);
        let out = dir.join("manifest.json");
        write_manifest_json(&out, &man).unwrap();
        assert!(out.is_file());
        let _ = fs::remove_dir_all(&dir);
    }
}
