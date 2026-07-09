#!/usr/bin/env Rscript
# Figure 5 --- Identification gate map for the outcome family.
# Rows = outcomes; columns = joint pre-trend p, lead-slope warning, post CI includes zero?,
# causal disposition. Communicates the audit discipline: some outcomes earn a causal reading
# (identified null), some do not (PT-fail descriptive; degenerate).

SELF <- "scripts/causality/figures/fig05_outcome_gate_map.R"
source(file.path(Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab"), "scripts/causality/figures/_fig_common.R"))
OUTD <- file.path(DATA, "analysis_r_out_h8")

fam <- c(twl_active_liquidity = "active liquidity", depth_1pct = "depth +/-1%",
         depth_2pct = "depth +/-2%", depth_5pct = "depth +/-5%", net_liq = "net liquidity",
         unique_lp_count = "unique LP count", lp_entry_count = "LP entry",
         lp_exit_count = "LP exit", position_duration_days = "position duration",
         jit_share_same_block = "JIT share", collect_amount1_native = "collected amount",
         vol0 = "token-0 volume", vol1 = "token-1 volume",
         lp_fee_income_native1 = "native LP fee income",
         lp_fee_income_per_active_liquidity = "fee income / active liq (ratio)")

inputs <- list()
rows <- lapply(names(fam), function(oc) {
  f <- file.path(OUTD, sprintf("summary_%s.json", oc))
  if (!file.exists(f)) return(data.frame(outcome = oc, label = fam[[oc]], jp = NA, slope = NA,
                                          czero = NA, disp = "degenerate / not informative", stringsAsFactors = FALSE))
  inputs[[length(inputs) + 1]] <<- f
  sm <- fromJSON(f); czero <- (sm$orig_ci_lo <= 0 && sm$orig_ci_hi >= 0)
  ptfail <- sm$joint_leads_p < 0.05; warn <- !ptfail && !is.na(sm$lead_slope_p) && sm$lead_slope_p < 0.10
  disp <- if (ptfail) "PT-fail; descriptive" else if (warn) "identified null (lead-slope caveat)" else "identified null"
  data.frame(outcome = oc, label = fam[[oc]], jp = sm$joint_leads_p, slope = sm$lead_slope_p,
             czero = czero, disp = disp, stringsAsFactors = FALSE)
})
R <- do.call(rbind, rows); print(R[, c("label", "jp", "czero", "disp")], digits = 3)

disp_col <- function(d) switch(d, "identified null" = PAL$null,
  "identified null (lead-slope caveat)" = PAL$caveat, "PT-fail; descriptive" = PAL$ptfail, PAL$degenerate)
tint <- function(hex, a = 0.22) { rgb <- col2rgb(hex) / 255; grDevices::rgb(rgb[1], rgb[2], rgb[3], alpha = a) }

draw <- function() {
  par(mar = c(0.5, 0.5, 3.4, 0.5), family = "sans", fg = PAL$ink)
  # group by disposition so the map reads at a glance (green block, then caveat, PT-fail, degenerate)
  rank <- c("identified null" = 1, "identified null (lead-slope caveat)" = 2,
            "PT-fail; descriptive" = 3, "degenerate / not informative" = 4)
  S <- R[order(rank[R$disp]), ]; nr <- nrow(S)
  plot(NA, xlim = c(0, 1), ylim = c(0, nr + 2.2), axes = FALSE, xlab = "", ylab = "")
  title(main = "Identification gate map: which outcomes earn a causal reading",
        adj = 0, cex.main = 1.2, font.main = 1)

  ## colour key (disposition -> tile colour)
  key <- list(c("identified null", PAL$null), c("lead-slope caveat", PAL$caveat),
              c("PT-fail; descriptive", PAL$ptfail), c("degenerate", PAL$degenerate))
  ky <- nr + 1.35; kx <- 0.02
  for (k in key) {
    rect(kx, ky - 0.30, kx + 0.028, ky + 0.30, col = k[[2]], border = NA)
    text(kx + 0.038, ky, k[[1]], adj = c(0, 0.5), cex = 0.82, col = PAL$sub)
    kx <- kx + 0.038 + strwidth(k[[1]], cex = 0.82) + 0.05
  }

  ## one row per outcome: name + a disposition-coloured tile carrying the pre-trend p
  tl <- 0.66; tr <- 0.80
  for (i in seq_len(nr)) {
    y <- nr - i + 1; r <- S[i, ]
    text(0.02, y, r$label, adj = c(0, 0.5), cex = 0.98, col = PAL$ink)
    rect(tl, y - 0.40, tr, y + 0.40, col = disp_col(r$disp), border = "white", lwd = 2)
    ptxt <- if (is.na(r$jp)) "n/a" else sprintf("%.3f", r$jp)
    text((tl + tr) / 2, y, ptxt, adj = c(0.5, 0.5), cex = 0.95, col = "white", font = 2)
  }
  fig_note(SELF)
}
save_fig("fig05_outcome_gate_map", 11.0, 5.6, draw)

write_manifest(
  "fig05_outcome_gate_map", SELF, inputs = inputs,
  claim  = "Outcomes are not treated equally: liquidity/depth/participation outcomes pass the pre-trend gate and receive an identified-null reading; token-1 volume and native fee income fail and stay descriptive; the fee-income ratio is degenerate.",
  role   = "audit discipline (identification routing)",
  caveat = "Gate: joint pre-trend p<0.05 => PT-fail/descriptive; lead-slope p<0.10 with a passing joint test => lead-slope caveat; post CI-includes-zero from the HonestDiD baseline interval. The ratio outcome is degenerate in the reconstructed panel.",
  extra  = list(rows = setNames(Map(function(jp, cz, d) list(joint_leads_p = jp, post_ci_incl_zero = cz, disposition = d),
                                     R$jp, R$czero, R$disp), R$outcome))
)
