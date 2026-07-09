#!/usr/bin/env Rscript
# Figure 4 --- Precision / minimum-detectable-effect (MDE) plot for the identified outcomes.
# For each outcome: aggregate post-period ATT interval (asinh, HonestDiD baseline), zero line,
# and the +/-MDE80 markers (effect magnitude detectable with 80% power at 5%). Grouped into
# (A) liquidity magnitude / depth and (B) LP participation / behavior. Makes the null honest:
# participation/behavior are sharply estimated; magnitude/depth are less precise.

SELF <- "scripts/causality/figures/fig04_precision_mde.R"
source(file.path(Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab"), "scripts/causality/figures/_fig_common.R"))
OUTD <- file.path(DATA, "analysis_r_out_h8")

spec <- list(
  A = c(twl_active_liquidity = "active liquidity", depth_1pct = "depth +/-1%",
        depth_2pct = "depth +/-2%", depth_5pct = "depth +/-5%",
        net_liq = "net liquidity", collect_amount1_native = "collected amount"),
  B = c(unique_lp_count = "unique LP count", lp_entry_count = "LP entry",
        lp_exit_count = "LP exit", position_duration_days = "position duration",
        jit_share_same_block = "JIT share"))

kM <- (qnorm(0.975) + qnorm(0.80))                                   # 2.802 : MDE80 multiplier
rows <- list(); inputs <- list()
for (grp in names(spec)) for (oc in names(spec[[grp]])) {
  f <- file.path(OUTD, sprintf("summary_%s.json", oc)); inputs[[length(inputs) + 1]] <- f
  sm <- fromJSON(f); lo <- sm$orig_ci_lo; hi <- sm$orig_ci_hi
  se <- (hi - lo) / (2 * 1.96); est <- (hi + lo) / 2
  rows[[length(rows) + 1]] <- data.frame(group = grp, outcome = oc, label = spec[[grp]][[oc]],
                                          lo = lo, hi = hi, est = est, se = se, mde80 = kM * se)
}
R <- do.call(rbind, rows)
print(R[, c("group", "label", "lo", "hi", "mde80")], digits = 3)

draw <- function() {
  par(mar = c(4.6, 11.5, 2.6, 1.4), oma = c(1.6, 0, 0, 0),
      family = "sans", col.axis = PAL$ink, col.lab = PAL$ink, fg = PAL$ink)
  # layout rows top->bottom: group A block, gap, group B block
  RA <- R[R$group == "A", ]; RB <- R[R$group == "B", ]
  yA <- rev(seq_len(nrow(RA))) + nrow(RB) + 1.4
  yB <- rev(seq_len(nrow(RB)))
  R$y <- c(yA, yB)
  xlim <- range(c(R$lo, R$hi, -R$mde80, R$mde80)) * 1.05
  plot(NA, xlim = xlim, ylim = c(0.3, max(yA) + 0.9), axes = FALSE,
       xlab = "aggregate post-period ATT interval (asinh scale)", ylab = "")
  abline(v = 0, col = PAL$zero, lwd = 1)
  for (i in seq_len(nrow(R))) {
    y <- R$y[i]
    segments(-R$mde80[i], y, R$mde80[i], y, col = PAL$grid, lwd = 5, lend = 1)   # +/-MDE80 band
    segments(R$lo[i], y, R$hi[i], y, col = PAL$control, lwd = 1.8)               # ATT CI
    points(R$est[i], y, pch = 19, col = PAL$control, cex = 0.9)
    axis(2, at = y, labels = R$label[i], las = 1, tick = FALSE, cex.axis = 0.82)
  }
  axis(1, cex.axis = 0.85)
  text(xlim[1], max(yA) + 0.7, "A. Liquidity magnitude / depth", adj = 0, cex = 0.82, font = 2, col = PAL$sub)
  text(xlim[1], max(yB) + 0.7, "B. LP participation / behavior", adj = 0, cex = 0.82, font = 2, col = PAL$sub)
  title(main = "Aggregate post-period intervals vs detectable effect size", adj = 0, cex.main = 1.05, font.main = 1)
  legend("bottomright", bty = "n", cex = 0.76,
         legend = c("aggregate post ATT 95% CI", "+/-MDE80 (80% power, 5%)"),
         col = c(PAL$control, PAL$grid), lwd = c(1.8, 5))
  fig_note(SELF)
}
save_fig("fig04_precision_mde", 10.5, 6.4, draw)

write_manifest(
  "fig04_precision_mde", SELF, inputs = inputs,
  claim  = "Participation and behavior outcomes are sharply estimated; liquidity magnitude and depth are less precise. The design rules out large short-run average responses more clearly than moderate capital reallocation.",
  role   = "precision / interpretation of the null",
  caveat = "Aggregate post ATT interval is the HonestDiD baseline (Mbar=0) on the asinh scale; SE is backed out from that interval and MDE80 = (z_0.975 + z_0.80) * SE = 2.80 * SE. Intervals are not precise zeros.",
  extra  = list(rows = setNames(Map(function(lo, hi, m) list(post_ci = c(round(lo, 4), round(hi, 4)),
                                     mde80 = round(m, 4)), R$lo, R$hi, R$mde80), R$outcome))
)
