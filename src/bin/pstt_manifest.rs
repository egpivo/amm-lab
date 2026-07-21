//! Hash explicit input paths into a new PSTT provenance manifest.
//!
//! Never edits historical `freeze_manifest_*.sha256` files.

use amm_lab::pstt::error::PsttError;
use amm_lab::pstt::manifest::{build_manifest, require_empty_output_dir, write_manifest_json};
use clap::Parser;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(about = "PSTT SHA-256 manifest writer (new artifacts only)")]
struct Args {
    /// Paths are relativized against this root in the manifest.
    #[arg(long)]
    base: PathBuf,
    /// Files to hash (repeatable).
    #[arg(long = "file", required = true)]
    files: Vec<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "pstt_manifest.json")]
    output_name: String,
    #[arg(long, default_value_t = false)]
    allow_nonempty_output: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(args: Args) -> Result<(), PsttError> {
    require_empty_output_dir(&args.output_dir, args.allow_nonempty_output)?;
    let abs_files: Vec<PathBuf> = args
        .files
        .iter()
        .map(|p| {
            if p.is_absolute() {
                p.clone()
            } else {
                args.base.join(p)
            }
        })
        .collect();
    let manifest = build_manifest(&args.base, &abs_files)?;
    let out = args.output_dir.join(&args.output_name);
    write_manifest_json(&out, &manifest)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "file_count": manifest.file_count,
            "total_bytes": manifest.total_bytes,
            "output": out,
            "generator": "pstt_manifest",
            "note": "Additive Rust provenance; does not modify historical freeze manifests."
        }))?
    );
    Ok(())
}
