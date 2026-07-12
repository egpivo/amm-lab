#!/usr/bin/env Rscript
# Figure 3 --- Main event-study paths: active liquidity and local depth.
# Event-time coefficients (asinh, ref k=-1) with token-pair CR1 95% intervals, primary +/-8-week
# window. Depth panel overlays +/-1% and +/-5% as thin grey paths. Annotates the joint lead-test
# p-value and the aggregate post-period HonestDiD interval. Intervals include zero throughout the
# post period: a non-detection at the design's resolution, not a precise zero.

SELF <- "scripts/causality/figures/fig03_main_event_paths.R"
source(file.path(Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab"), "scripts/causality/figures/_fig_common.R"))
OUTD <- file.path(DATA, "analysis_r_out_h8")

es_read <- function(oc) {
  es <- read.csv(file.path(OUTD, sprintf("event_study_%s.csv", oc)), stringsAsFactors = FALSE)
  es <- rbind(es[, c("rel", "beta", "se")], data.frame(rel = -1, beta = 0, se = NA))  # add reference k=-1
  es <- es[order(es$rel), ]
  sm <- fromJSON(file.path(OUTD, sprintf("summary_%s.json", oc)))
  list(es = es, sm = sm)
}

twl <- es_read("twl_active_liquidity")
d2  <- es_read("depth_2pct")
d1  <- es_read("depth_1pct"); d5 <- es_read("depth_5pct")

inputs <- as.list(file.path(OUTD, c(
  sprintf("event_study_%s.csv", c("twl_active_liquidity", "depth_2pct", "depth_1pct", "depth_5pct")),
  sprintf("summary_%s.json",    c("twl_active_liquidity", "depth_2pct", "depth_1pct", "depth_5pct")))))

# Panels rendered separately and composed as LaTeX subfigures (full canvas, large fonts).
panel <- function(main, sm, title, grays = list()) {
  par(mar = c(4.9, 5.1, 1.5, 1.2), family = "sans", col.axis = PAL$ink, col.lab = PAL$ink, fg = PAL$ink)
  es <- main$es; ci_lo <- es$beta - 1.96 * es$se; ci_hi <- es$beta + 1.96 * es$se
  yl <- range(c(ci_lo, ci_hi, unlist(lapply(grays, function(g) g$es$beta)), 0), na.rm = TRUE)
  yl <- yl + c(-0.15, 0.15) * diff(yl)
  plot(NA, xlim = c(-8.4, 8.4), ylim = yl, xlab = "event time (weeks relative to activation)",
       ylab = "coefficient (asinh)", axes = FALSE, cex.lab = 1.2)
  hgrid(pretty(yl)); axis(1, at = seq(-8, 8, 2), cex.axis = 1.05); axis(2, las = 1, cex.axis = 1.05)
  abline(h = 0, col = PAL$zero, lwd = 1); abline(v = -0.5, col = PAL$zero, lwd = 1, lty = 3)
  for (g in grays) lines(g$es$rel, g$es$beta, col = PAL$grid, lwd = 2)
  segments(es$rel, ci_lo, es$rel, ci_hi, col = PAL$control, lwd = 2)
  lines(es$rel, es$beta, col = PAL$control, lwd = 1.6)
  points(es$rel, es$beta, pch = 19, col = PAL$control, cex = 1.0)
  points(-1, 0, pch = 21, bg = "white", col = PAL$ink, cex = 1.2)                 # reference marker
  title(main = title, adj = 0, cex.main = 1.2, font.main = 1)
  txt <- sprintf("joint pre-trend p = %.2f\naggregate post CI [%.2f, %.2f]",
                 sm$joint_leads_p, sm$orig_ci_lo, sm$orig_ci_hi)
  legend("bottomleft", legend = txt, bty = "n", cex = 1.0, text.col = PAL$ink)
  if (length(grays)) legend("topright", legend = c("+/-2% (primary)", "+/-1%, +/-5%"),
                            col = c(PAL$control, PAL$grid), lwd = 2, bty = "n", cex = 0.95)
}

save_fig("fig03_paths_A", 7.4, 6.0, function() panel(twl, twl$sm, "Time-weighted active liquidity"))
save_fig("fig03_paths_B", 7.4, 6.0, function() panel(d2, d2$sm, "Local depth (+/-2% primary)", grays = list(d1, d5)))

write_manifest(
  "fig03_main_event_paths", SELF, inputs = inputs,
  claim  = "The main empirical result: event-study paths for active liquidity and local depth show no detectable post-activation departure; intervals include zero throughout the post period.",
  role   = "MAIN causal result (identified LP-supply / depth null)",
  caveat = "Intervals include zero throughout the post period. This is a non-detection at the design's resolution, not a precise zero. CR1 token-pair-clustered 95% intervals shown; restricted wild cluster bootstrap p-values per coefficient are in the event_study CSVs; the annotated aggregate post interval is the HonestDiD baseline (Mbar=0).",
  extra  = list(twl = list(joint_leads_p = twl$sm$joint_leads_p, post_ci = c(twl$sm$orig_ci_lo, twl$sm$orig_ci_hi)),
                depth_2pct = list(joint_leads_p = d2$sm$joint_leads_p, post_ci = c(d2$sm$orig_ci_lo, d2$sm$orig_ci_hi)))
)
