#!/usr/bin/env Rscript
# Figure 2 --- Matched-overlap balance and support.
# Panel A: distribution of log pre-period realized fee revenue for treated main, control
#          reservoir, and reuse-weighted matched controls (caliper 0.5). Shows overlap.
# Panel B: signed-SMD love plot (before vs after matching) for the pre-period covariates.
#          The point is honesty: residual imbalance on the fee-revenue selection margin
#          remains; identification rests on the event-study gates, not level balance.

SELF <- "scripts/causality/figures/fig02_overlap_balance.R"
source(file.path(Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab"), "scripts/causality/figures/_fig_common.R"))

fr_path  <- file.path(DATA, "feerev_panelvars.csv")
mp_path  <- file.path(DATA, "matched_pairs_cal0.5.json")
pan_path <- file.path(DATA, "panel_weekly_rust.csv")
mf_path  <- file.path(DATA, "match_feasibility.json")
fac_path <- file.path(DATA, "ckpt_factory.json")
tok_path <- file.path(DATA, "ckpt_tokens.json")
CANON <- "0x1f98431c8ad98523631ae4a59f267346ea31f984"

fr <- read.csv(fr_path, stringsAsFactors = FALSE, colClasses = c(pool = "character"))
fr$fr12 <- suppressWarnings(as.numeric(fr$fr12_usd)); fr$sw12 <- suppressWarnings(as.numeric(fr$sw12))
mp <- fromJSON(mp_path); mf <- fromJSON(mf_path)
fac <- fromJSON(fac_path); tok <- fromJSON(tok_path)

treated_main <- fr$pool[fr$treated == "1" & fr$old == "1" & fr$covered == "1" & fr$fr12 > 0]      # 868
reservoir    <- fr$pool[fr$treated == "0" & fr$old == "1" & fr$covered == "1" & fr$fr12 > 0]      # 669

## reconstruct the frozen canon_reservoir (622): factory==CANON & exposure<=0.25 (caliper_rematch.py)
treated_all <- fr$pool[fr$treated == "1"]
tpairs <- unique(vapply(treated_all, function(p){ t <- tok[[p]]; t <- t[nzchar(t)]; if (length(t) >= 2) paste(sort(tolower(t)), collapse = "|") else "" }, character(1)))
ttok <- unique(tolower(unlist(lapply(treated_all, function(p){ t <- tok[[p]]; t[nzchar(t)] }))))
exposure <- function(p){ t <- tok[[p]]; t <- tolower(t[nzchar(t)]); if (length(t) < 2) return(1)
  if (paste(sort(t), collapse = "|") %in% tpairs) return(1)
  if (any(t %in% ttok)) return(0.125); 0 }
canon_reservoir <- reservoir[vapply(reservoir, function(p) isTRUE(tolower(fac[[p]]) == CANON) && exposure(p) <= 0.25, logical(1))]

tm           <- mp$treated                                   # 786 matched treated (frozen SMD group)
matched_ctrl <- sort(unique(unlist(mp$controls)))            # 314 controls used (unweighted, per frozen)
cat(sprintf("treated_main=%d reservoir=%d canon_reservoir=%d matched_treated(tm)=%d ctrl_used=%d\n",
            length(treated_main), length(reservoir), length(canon_reservoir), length(tm), length(matched_ctrl)))

## ---- pre-period covariates from the frozen panel (rel<0) ----
gi <- grid_index(); t0i <- gi[["2025-51"]]
need <- unique(c(treated_main, canon_reservoir, matched_ctrl))
p <- read.csv(pan_path, stringsAsFactors = FALSE,
              colClasses = c(pool = "character", unit_role = "character", week = "character"))
p <- p[p$pool %in% need & p$week %in% names(gi), ]
p$rel <- gi[p$week] - t0i; p <- p[p$rel < 0, ]                        # PRE-PERIOD ONLY
sp_ord <- order(p$pool, p$rel)
volat  <- tapply(seq_len(nrow(p)), p$pool, function(ix){ d <- p[ix, ][order(p$rel[ix]), ]
  pr <- log((d$vol1 + 1) / (d$vol0 + 1)); pr <- pr[is.finite(pr)]
  if (length(pr) >= 5) sd(diff(pr)) else NA_real_ })
vol_mean   <- tapply(p$vol0 + p$vol1, p$pool, mean, na.rm = TRUE)
depth_mean <- tapply(p$depth_2pct, p$pool, mean, na.rm = TRUE)
jit_mean   <- tapply(p$jit_share_same_block, p$pool, mean, na.rm = TRUE)

# covariate value vectors keyed by pool (frozen transforms: fee/swap = natural log; others log1p)
V <- list(
  logfr    = log(setNames(fr$fr12, fr$pool)),
  logsw    = log(setNames(fr$sw12, fr$pool)),
  logvol   = log1p(vol_mean), volat = volat,
  logdepth = log1p(depth_mean), jit = jit_mean)
covs <- c(logfr = "log pre-period fee revenue", logsw = "log pre-period swap count",
          logvol = "log pre-period volume", volat = "volatility (proxy)",
          logdepth = "log local depth (+/-2%)", jit = "JIT share")

## frozen SMD convention (caliper_rematch.py smd()): population sd, pooled per comparison, unweighted
popsd <- function(x){ x <- x[is.finite(x)]; sqrt(mean((x - mean(x))^2)) }
smd1  <- function(vals, A, B){ a <- vals[A]; b <- vals[B]; a <- a[is.finite(a)]; b <- b[is.finite(b)]
  sp <- sqrt((popsd(a)^2 + popsd(b)^2) / 2); (mean(a) - mean(b)) / sp }
# Coverage note: the frozen estimation panel holds all 314 used controls but only a partial
# slice of the 622-pool reservoir, so a faithful BEFORE-matching SMD over the full reservoir is
# only available for the feerev covariates (fee revenue, swap count). Panel-derived covariates
# (volume/depth/JIT/volatility) show POST-match estimation-sample balance only (tm vs used).
feerev_covs <- c("logfr", "logsw")
smd <- data.frame(var = names(covs), label = unname(covs), before = NA_real_, after = NA_real_)
for (v in names(covs)) {
  smd$after[smd$var == v]  <- smd1(V[[v]], tm, matched_ctrl)                                # always valid
  if (v %in% feerev_covs) smd$before[smd$var == v] <- smd1(V[[v]], tm, canon_reservoir)     # full reservoir
}
print(smd, digits = 3)
cat(sprintf("artifact cross-check: logfr before/after %.3f/%.3f (mine %.3f/%.3f) ; logsw %.3f/%.3f (mine %.3f/%.3f)\n",
            mf$smd_logfr_before, mf$smd_logfr_after, smd$before[1], smd$after[1],
            mf$smd_logsw_before, mf$smd_logsw_after, smd$before[2], smd$after[2]))

# Panels rendered separately and composed as LaTeX subfigures (full canvas, large fonts).

## ---- Panel A: log fee-revenue densities (natural log; frozen groups, unweighted) ----
pA <- function() {
  par(mar = c(4.9, 5.1, 1.5, 1.2), family = "sans", col.axis = PAL$ink, col.lab = PAL$ink, fg = PAL$ink)
  gv <- function(set) { x <- V$logfr[set]; x[is.finite(x)] }
  dt <- density(gv(treated_main)); dr <- density(gv(canon_reservoir)); dm <- density(gv(matched_ctrl))
  xr <- range(dt$x, dr$x, dm$x); yr <- c(0, max(dt$y, dr$y, dm$y) * 1.05)
  plot(NA, xlim = xr, ylim = yr, xlab = "log pre-period fee revenue (natural log, USD)",
       ylab = "density", axes = FALSE, cex.lab = 1.2)
  axis(1, cex.axis = 1.05); axis(2, las = 1, cex.axis = 1.05)
  lines(dr, col = PAL$control, lwd = 2.4); lines(dm, col = PAL$matched, lwd = 2.4)
  lines(dt, col = PAL$treated, lwd = 2.4)
  legend("topright", bty = "n", cex = 1.0, lwd = 2.4,
         col = c(PAL$treated, PAL$control, PAL$matched),
         legend = c(sprintf("treated main (n=%d)", length(treated_main)),
                    sprintf("control reservoir (n=%d)", length(canon_reservoir)),
                    sprintf("matched controls (n=%d)", length(matched_ctrl))))
}

## ---- Panel B: signed-SMD love plot (before/after for matching covariates; post-match for rest) ----
pB <- function() {
  par(mar = c(4.9, 0.8, 1.5, 1.2), family = "sans", col.axis = PAL$ink, col.lab = PAL$ink, fg = PAL$ink)
  s <- smd[order(is.na(smd$before), abs(smd$after)), ]; ny <- nrow(s); ys <- rev(seq_len(ny))
  xlim <- c(min(0, s$before, s$after, na.rm = TRUE) - 0.05, max(s$before, s$after, na.rm = TRUE) + 0.10)
  plot(NA, xlim = xlim, ylim = c(0.5, ny + 0.9), axes = FALSE,
       xlab = "standardized mean difference (treated - control)", ylab = "", cex.lab = 1.2)
  abline(v = 0, col = PAL$zero, lwd = 1)
  for (thr in c(0.1, 0.25)) abline(v = c(-thr, thr), col = PAL$grid, lwd = 0.8, lty = 2)
  for (i in seq_len(ny)) {
    y <- ys[i]
    if (is.finite(s$before[i])) {
      segments(s$before[i], y, s$after[i], y, col = PAL$grid, lwd = 1.8)
      points(s$before[i], y, pch = 21, bg = "white", col = PAL$sub, cex = 1.5)
    }
    points(s$after[i], y, pch = 19, col = PAL$treated, cex = 1.5)
    lab <- if (is.finite(s$before[i])) s$label[i] else paste0(s$label[i], "  (post-match only)")
    text(xlim[1], y + 0.34, lab, adj = 0, cex = 0.98, col = PAL$ink)
  }
  axis(1, cex.axis = 1.05)
  legend("bottomright", bty = "n", cex = 0.95, pch = c(21, 19), pt.bg = c("white", NA),
         col = c(PAL$sub, PAL$treated), legend = c("before matching", "after matching"))
  text(xlim[1], 0.7, "dashed: |SMD| = 0.10, 0.25", col = PAL$sub, cex = 0.85, adj = 0)
}

save_fig("fig02_balance_A", 7.4, 6.0, pA)
save_fig("fig02_balance_B", 7.4, 6.0, pB)

write_manifest(
  "fig02_overlap_balance", SELF,
  inputs = list(fr_path, mp_path, pan_path, mf_path, fac_path, tok_path),
  claim  = "The matched-overlap design achieves support but not level balance: matching narrows the fee-revenue gap (SMD 1.19 -> 0.84) yet substantial residual imbalance remains on the selection margin.",
  role   = "design transparency (not a balance-as-identification claim)",
  caveat = "SMDs reproduce the frozen convention exactly (caliper_rematch.py: matched treated n=786 vs reservoir n=622 / used controls n=314; natural-log fee/swap; population sd pooled per comparison; unweighted). Added covariates (volume/depth/JIT/volatility) use the same convention; volatility is a depth-free proxy (sd of weekly log implied-price change). Pool age is a binary eligibility filter (old>=52wk) with no matched-sample variation, so it is excluded. Identification relies on conditional parallel trends / event-study gates, not level balance.",
  extra  = list(smd = setNames(Map(function(b, a) list(before = round(b, 4), after = round(a, 4)),
                                    smd$before, smd$after), smd$var),
                artifact_smd = list(logfr = c(mf$smd_logfr_before, mf$smd_logfr_after),
                                    logsw = c(mf$smd_logsw_before, mf$smd_logsw_after)),
                n = list(treated_main = length(treated_main), canon_reservoir = length(canon_reservoir),
                         matched_treated = length(tm), matched_controls = length(matched_ctrl)))
)
