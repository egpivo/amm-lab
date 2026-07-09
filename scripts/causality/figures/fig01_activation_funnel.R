#!/usr/bin/env Rscript
# Figure 1 --- Treatment activation and sample funnel.
# Panel A: weekly SetFeeProtocol activation counts by protocol-share intensity group, from the
#          reconstructed on-chain events (setfeeprotocol_events.csv). Shows the real governance
#          burst rather than a schematic timeline.
# Panel B: the frozen design funnel from 2,638 canonical treated pools to the 779-pool
#          estimation sample. Terminal counts (868/786/779) are cross-checked against
#          match_stats/summary artifacts.

SELF <- "scripts/causality/figures/fig01_activation_funnel.R"
source(file.path(Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab"), "scripts/causality/figures/_fig_common.R"))

ev_path  <- file.path(DATA, "setfeeprotocol_events.csv")
inv_path <- file.path(DATA, "activation_inventory.json")
ms_path  <- file.path(DATA, "match_stats_cal0.5.json")
sm_path  <- file.path(DATA, "analysis_r_out_h8", "summary_twl_active_liquidity.json")

ev  <- read.csv(ev_path, stringsAsFactors = FALSE, colClasses = c(pool = "character", tx = "character"))
inv <- fromJSON(inv_path)
ms  <- fromJSON(ms_path)
sm  <- fromJSON(sm_path)

## intensity group from tier (canonical mapping; activation_inventory$intensity_by_tier)
ev$intensity <- ifelse(ev$fee_tier %in% c(100, 500), "1/4 share (1bp, 5bp)",
                ifelse(ev$fee_tier %in% c(3000, 10000), "1/6 share (30bp, 100bp)", NA))
ev <- ev[!is.na(ev$intensity), ]
ev$date <- as.POSIXct(ev$utc, tz = "UTC", format = "%Y-%m-%d %H:%M")
ev$wk   <- as.Date(cut(as.Date(ev$date), breaks = "week", start.on.monday = TRUE))

wks   <- sort(unique(ev$wk))
grpsA <- c("1/4 share (1bp, 5bp)", "1/6 share (30bp, 100bp)")
cnt   <- sapply(grpsA, function(g) sapply(wks, function(w) sum(ev$wk == w & ev$intensity == g)))
cnt   <- matrix(cnt, nrow = length(wks), dimnames = list(as.character(wks), grpsA))
burst_wk <- wks[which.max(rowSums(cnt))]

## funnel (frozen design facts; terminal three cross-checked against artifacts)
fun_lab <- c("Canonical v3 treated pools",
             "Active in pre-proposal census week",
             ">=52-week pre-period proxy",
             "Stable / ETH / BTC numeraire leg",
             "Positive pre-period fee revenue",
             "Matched at 0.5 caliper",
             "Estimation sample (+/-8 wk, singletons removed)")
fun_val <- c(inv$canonical_v3_treated_pools, 1426, 961, 870,
             ms$n_treated_main, ms$n_matched, sm$matched_treated)
stopifnot(fun_val[5] == 868, fun_val[6] == 786, fun_val[7] == 779)  # artifact cross-check

# Panels are rendered to separate files and composed as LaTeX subfigures, so each panel gets
# the full canvas and large fonts.

## ---- Panel A: weekly activation, log axis (burst dwarfs stragglers) ----
pA <- function() {
  par(mar = c(5.4, 5.2, 1.2, 1.2), family = "sans", col.axis = PAL$ink, col.lab = PAL$ink, fg = PAL$ink)
  h <- log10(t(cnt) + 1)
  bp <- barplot(h, beside = TRUE, border = NA, col = c(PAL$k4, PAL$k6),
                ylim = c(0, max(h) * 1.18), axes = FALSE, names.arg = rep("", length(wks)),
                ylab = "activations per week (log scale)", cex.lab = 1.2)
  yt <- c(0, 1, 10, 100, 1000); axis(2, at = log10(yt + 1), labels = yt, las = 1, cex.axis = 1.05)
  xm <- colMeans(bp)
  lab <- format(wks, "%b %d"); show <- seq(1, length(wks), by = max(1, round(length(wks) / 8)))
  axis(1, at = xm[show], labels = lab[show], las = 2, cex.axis = 1.0, tick = FALSE)
  text(xm[1], max(h) * 0.58, sprintf("governance burst\n%s\n(n=%s in this week)",
       inv$activation_burst_utc, format(sum(cnt[as.character(burst_wk), ]), big.mark = ",")),
       cex = 1.05, col = PAL$ink, pos = 4, offset = 0)
  legend("topright", legend = grpsA, fill = c(PAL$k4, PAL$k6), border = NA, bty = "n", cex = 1.1)
}

## ---- Panel B: sample funnel (airy: thin bars, clear label gap) ----
pB <- function() {
  par(mar = c(1.0, 0.8, 1.2, 1.4), family = "sans", fg = PAL$ink)
  n <- length(fun_val); ys <- rev(seq_len(n)); w <- fun_val / max(fun_val)
  plot(NA, xlim = c(0, 1.26), ylim = c(0.4, n + 0.85), axes = FALSE, xlab = "", ylab = "")
  for (i in seq_len(n)) {
    y <- ys[i]
    rect(0, y - 0.18, w[i], y + 0.18, col = PAL$bar, border = NA)
    text(0.004, y + 0.42, fun_lab[i], adj = 0, cex = 1.08, col = PAL$ink)
    text(w[i] + 0.015, y, format(fun_val[i], big.mark = ","), adj = 0, cex = 1.12, col = PAL$ink)
  }
}

save_fig("fig01_activation_A", 7.6, 6.0, pA)
save_fig("fig01_activation_B", 7.6, 6.0, pB)

write_manifest(
  "fig01_activation_funnel", SELF,
  inputs = list(ev_path, inv_path, ms_path, sm_path),
  claim  = "The protocol-fee shock is a real, reconstructed on-chain event: a governance burst activating ~2,638 canonical v3 pools, funneled by the frozen design to a 779-pool estimation sample.",
  role   = "context / data provenance",
  caveat = "Funnel steps 2-4 (census week, >=52-week proxy, numeraire leg) are frozen design counts; the terminal three (868/786/779) are cross-checked against match_stats and the twl summary. Intensity groups use the canonical tier->share map.",
  extra  = list(burst_week = as.character(burst_wk),
                burst_count = as.integer(sum(cnt[as.character(burst_wk), ])),
                n_events_total = nrow(ev), funnel = setNames(as.list(fun_val), fun_lab))
)
