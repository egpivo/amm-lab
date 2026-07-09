# _fig_common.R  --- shared style + provenance for Paper C real-data figures.
# Every figure sources this. It fixes the muted palette, the frozen week grid (identical to
# analysis_es.R), and writes a sidecar JSON manifest recording input artifact paths + SHA256,
# git commit, script, timestamp, and the frozen design constants (window / caliper / transform /
# inference). Figures are DESCRIPTIVE renderings of frozen artifacts; nothing here re-estimates.

suppressMessages({ library(jsonlite); library(digest) })

ROOT   <- Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA   <- Sys.getenv("AMMLAB_DATA", file.path(ROOT, "data/causality"))
FIGDIR <- Sys.getenv("AMMLAB_FIGDIR", file.path(ROOT, ".local/amm_paper_c/figures"))
dir.create(FIGDIR, showWarnings = FALSE, recursive = TRUE)

## ---- muted palette (calm, low-saturation; consistent across all figures) ----
PAL <- list(
  ink = "#2B2B2B", sub = "#5F6368", grid = "#D9D7D1", zero = "#7A7A7A",
  treated = "#8E3B3B", control = "#4C6A91", matched = "#6A8F73",
  k4 = "#4C6A91", k6 = "#B7905E",                 # intensity 1/4 vs 1/6
  null = "#6A8F73", ptfail = "#B65C5C", caveat = "#B7905E", degenerate = "#9AA0A6",
  bar = "#4C6A91"
)

## ---- frozen design constants (do NOT change; must match the estimation pipeline) ----
DESIGN <- list(
  window    = "+/-8 weeks (primary)",
  ref_period = -1L,
  t0        = "2025-51",
  caliper   = 0.5,
  transform = "asinh",
  inference = "token-pair CR1 cluster-robust; restricted wild cluster bootstrap (fwildclusterboot); HonestDiD relative-magnitude bounds"
)

## ---- frozen week grid index (identical to analysis_es.R / Rust WeekGrid) ----
grid_index <- function() {
  b0 <- as.integer(as.POSIXct("2024-01-01 00:00:00", tz = "UTC"))
  b1 <- as.integer(as.POSIXct("2026-06-30 23:59:59", tz = "UTC"))
  ts <- seq(b0, b1, by = 86400)
  labs <- sort(unique(format(as.POSIXct(ts, origin = "1970-01-01", tz = "UTC"), "%Y-%W")))
  setNames(seq_along(labs) - 1L, labs)
}

## ---- provenance ----
git_hash <- function() tryCatch(trimws(system2("git", c("-C", ROOT, "rev-parse", "HEAD"),
                                                stdout = TRUE, stderr = FALSE)),
                                error = function(e) NA_character_)
sha256f  <- function(p) if (file.exists(p)) digest(file = p, algo = "sha256") else NA_character_
relp     <- function(p) sub(paste0(normalizePath(ROOT), "/"), "", normalizePath(p, mustWork = FALSE))

write_manifest <- function(fig, script, inputs, claim = NULL, role = NULL, caveat = NULL, extra = list()) {
  m <- c(list(
    figure = fig, script = script,
    generated_utc = format(as.POSIXct(Sys.time()), tz = "UTC", "%Y-%m-%dT%H:%M:%SZ"),
    git_commit = git_hash(),
    window = DESIGN$window, ref_period = DESIGN$ref_period, t0 = DESIGN$t0,
    caliper = DESIGN$caliper, transform = DESIGN$transform, inference = DESIGN$inference,
    claim = claim, role = role, caveat = caveat,
    inputs = lapply(inputs, function(p) list(path = relp(p), sha256 = sha256f(p)))
  ), extra)
  write_json(m, file.path(FIGDIR, paste0(fig, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = 8)
  invisible(m)
}

## ---- render both PDF and PNG from one deterministic draw() ----
## Larger base pointsize (zen typography: readable without zoom); white background for
## embedding in the (white) paper page rather than the zen cream canvas.
save_fig <- function(fig, w, h, draw, pointsize = 15) {
  pdf(file.path(FIGDIR, paste0(fig, ".pdf")), width = w, height = h, pointsize = pointsize); draw(); dev.off()
  png(file.path(FIGDIR, paste0(fig, ".png")), width = round(w * 150), height = round(h * 150),
      res = 150, pointsize = pointsize); draw(); dev.off()
  cat(sprintf("wrote %s.{pdf,png}\n", file.path(FIGDIR, fig)))
}

## ---- provenance now lives ONLY in the sidecar JSON (no in-plot footnote) ----
fig_note <- function(script) invisible(NULL)

## ---- light horizontal gridlines helper ----
hgrid <- function(at) abline(h = at, col = PAL$grid, lwd = 0.6)
