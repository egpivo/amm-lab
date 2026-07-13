#!/usr/bin/env Rscript
# Paper C causal-inference driver (R base). One outcome, full gated pipeline:
#   matched-overlap sample -> outcome transform (Amendment 012) -> two-way FE event study
#   (fixest) -> joint pre-trend Wald + lead linear-trend -> WCR wild cluster bootstrap
#   (fwildclusterboot) -> HonestDiD relative-magnitude sensitivity.
# Reports point estimates + cluster-robust + WCR; the Rust `event_study` reproduces the
# point estimates as a parity cross-check. NOTHING here is asserted for the manuscript until
# the full outcome family + robustness are reviewed.
#
# Usage: Rscript analysis_es.R --outcome twl_active_liquidity [--transform asinh]
#        [--t0 2025-51] [--horizon 12] [--reps 9999] [--panel PATH] [--out DIR]

suppressMessages({ library(fixest); library(jsonlite) })
have_boot   <- requireNamespace("fwildclusterboot", quietly = TRUE)
have_honest <- requireNamespace("HonestDiD", quietly = TRUE)

## ---- args ----
args <- commandArgs(trailingOnly = TRUE)
getopt <- function(k, d) { i <- match(k, args); if (!is.na(i) && i < length(args)) args[i+1] else d }
ROOT    <- Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA    <- Sys.getenv("AMMLAB_DATA", file.path(ROOT, "data/causality"))
outcome <- getopt("--outcome", "twl_active_liquidity")
transf  <- getopt("--transform", "asinh")
t0lab   <- getopt("--t0", "2025-51")
H       <- as.integer(getopt("--horizon", "12"))
condtr  <- getopt("--condtrends", "none")   # none | stratum (Amendment 013 secondary diagnostic)
reps    <- as.integer(getopt("--reps", "9999"))
panel   <- getopt("--panel", file.path(DATA, "panel_weekly_rust.csv"))
outdir  <- getopt("--out", file.path(DATA, "analysis_r_out"))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

## ---- frozen week grid index (matches Rust WeekGrid / Python) ----
grid_index <- function() {
  b0 <- as.integer(as.POSIXct("2024-01-01 00:00:00", tz = "UTC"))
  b1 <- as.integer(as.POSIXct("2026-06-30 23:59:59", tz = "UTC"))
  ts <- seq(b0, b1, by = 86400)
  labs <- unique(format(as.POSIXct(ts, origin = "1970-01-01", tz = "UTC"), "%Y-%W"))
  setNames(seq_along(sort(labs)) - 1L, sort(labs))
}
gi <- grid_index()
stopifnot(t0lab %in% names(gi)); t0i <- gi[[t0lab]]

## ---- design metadata ----
tok <- fromJSON(file.path(DATA, "ckpt_tokens.json"))
cluster_of <- vapply(tok, function(p) { s <- sort(tolower(p[nzchar(p)])); if (length(s)) paste(s, collapse = "-") else "na" }, character(1))
# tier + pair-class per pool (matching strata) for the conditional-trends diagnostic
fr <- read.csv(file.path(DATA, "feerev_panelvars.csv"), stringsAsFactors = FALSE,
               colClasses = c(pool = "character"))
tier_of <- setNames(as.character(fr$tier), fr$pool)
class_of <- setNames(as.character(fr$class), fr$pool)
mpfile  <- getopt("--matched-pairs", file.path(DATA, "matched_pairs.json"))  # caliper sensitivity
mp <- fromJSON(mpfile)                                     # data.frame: treated, controls(list), ...
treated_set <- unique(mp$treated)
ctrl_counts <- table(unlist(mp$controls))
weight_of <- c(setNames(rep(1, length(treated_set)), treated_set),
               setNames(as.numeric(ctrl_counts), names(ctrl_counts)))

## ---- panel -> analysis frame ----
# NB: R 4.x read.csv parses "0x..." as hexadecimal numeric -> pool addresses would become
# huge numbers and never match. Force the address / label columns to character.
d <- read.csv(panel, stringsAsFactors = FALSE,
              colClasses = c(pool = "character", unit_role = "character", week = "character"))
d <- d[d$week %in% names(gi), ]                            # frozen window (drops 2023-52)
d <- d[d$pool %in% names(weight_of), ]                     # matched-overlap units
d$treated <- as.integer(d$pool %in% treated_set)
d$w <- weight_of[d$pool]
d$cluster_key <- ifelse(!is.na(cluster_of[d$pool]), cluster_of[d$pool], "na")
d$rel <- gi[d$week] - t0i
d <- d[abs(d$rel) <= H, ]                                  # windowed event study
# heterogeneity subsetting (Amendment 019): restrict to a pair-class and/or fee tier stratum
fclass <- getopt("--filter-class", ""); ftier <- getopt("--filter-tier", "")
if (nzchar(fclass)) d <- d[class_of[d$pool] == fclass, ]
if (nzchar(ftier))  d <- d[tier_of[d$pool] == ftier, ]
# depth-implied slippage PROXY (Amendment 019): mean token-1 trade size per unit local depth
# (approx local price impact). NOT observed slippage; a depth-implied proxy. Demote if unstable.
if (outcome == "slippage_proxy") {
  d$slippage_proxy <- (d$vol1 / pmax(d$swaps, 1)) / pmax(d$depth_2pct, 1e-9)
}
if (!outcome %in% names(d)) stop(sprintf("unknown outcome %s", outcome))
d <- d[is.finite(d[[outcome]]), ]
d$yt <- switch(transf,
               asinh = asinh(d[[outcome]]),
               log1p = log1p(pmax(d[[outcome]], 0)),
               d[[outcome]])
d$rel <- factor(d$rel)
d$rel <- relevel(d$rel, ref = "-1")                        # omit event-time -1
# boottest.fixest needs the FE / cluster vars as factors (it chokes on character FE formulas).
d$pool <- factor(d$pool); d$week <- factor(d$week); d$cluster_key <- factor(d$cluster_key)
# matching-stratum (tier x pair-class) for the conditional-parallel-trends diagnostic (A013)
d$stratum <- factor(paste(tier_of[as.character(d$pool)], class_of[as.character(d$pool)], sep = ":"))
fe_str <- if (condtr == "stratum") "pool + stratum^week" else "pool + week"

n_t <- length(unique(d$pool[d$treated == 1])); n_c <- length(unique(d$pool[d$treated == 0]))
cat(sprintf("matched-overlap: %d treated + %d control | %d pool-weeks | clusters %d | rel |<=%d| | %s [%s] | t0 %s\n",
            n_t, n_c, nrow(d), length(unique(d$cluster_key)), H, outcome, transf, t0lab))

## ---- two-way FE event study (fixest), cluster at token-pair, freq weights ----
# Purge singleton pool/week fixed effects BEFORE fitting: fwildclusterboot::boottest refuses
# a fixest object that dropped singletons internally, so we iterate feols -> keep obs(m) until
# the estimation sample is stable (no more singletons to remove).
es_fml <- as.formula(paste0("yt ~ i(rel, treated, ref = \"-1\") | ", fe_str))
fit_es <- function(dat) feols(es_fml, data = dat, weights = ~w, cluster = ~cluster_key)
repeat {
  m <- fit_es(d)
  used <- fixest::obs(m)
  if (length(used) == nrow(d)) break
  d <- droplevels(d[used, , drop = FALSE])
}
if (nrow(d) < 1) stop("empty estimation sample after singleton purge")
ct <- coeftable(m); cf <- coef(m); V <- vcov(m)
nm <- names(cf)
rel_of <- function(s) { r <- regmatches(s, regexpr("-?[0-9]+", s)); if (length(r)) as.integer(r) else NA_integer_ }
es_ix <- which(!is.na(vapply(nm, rel_of, integer(1))) & grepl("treated", nm))
rels  <- vapply(nm[es_ix], rel_of, integer(1))
ord   <- order(rels); es_ix <- es_ix[ord]; rels <- rels[ord]

tabmat <- data.frame(rel = rels, coef = nm[es_ix],
                     beta = cf[es_ix], se = ct[es_ix, "Std. Error"],
                     crv1_p = ct[es_ix, "Pr(>|t|)"], row.names = NULL)

## ---- joint pre-trend tests ----
lead_ix <- es_ix[rels < 0]; lead_rel <- rels[rels < 0]
bL <- cf[lead_ix]; VL <- V[lead_ix, lead_ix, drop = FALSE]
Wj <- as.numeric(t(bL) %*% MASS::ginv(VL) %*% bL)
joint_p <- pchisq(Wj, df = length(lead_ix), lower.tail = FALSE)
g <- lead_rel - mean(lead_rel); gamma <- sum(g * bL); gvar <- as.numeric(t(g) %*% VL %*% g)
slope_p <- if (gvar > 0) 2 * pnorm(abs(gamma / sqrt(gvar)), lower.tail = FALSE) else NA
max_abs_pre <- max(abs(bL))

## ---- WCR (restricted wild cluster bootstrap) per event-time coef ----
tabmat$wcr_p <- NA_real_
if (have_boot) {
  set.seed(42); suppressWarnings(try(dqrng::dqset.seed(42), silent = TRUE))  # reproducible WCR
  for (i in seq_len(nrow(tabmat))) {
    pj <- tryCatch({
      bt <- fwildclusterboot::boottest(m, param = tabmat$coef[i], clustid = ~cluster_key,
                                       B = reps, type = "rademacher", impose_null = TRUE)
      bt$p_val
    }, error = function(e) NA_real_)
    tabmat$wcr_p[i] <- pj
  }
}

## ---- HonestDiD relative-magnitude sensitivity (avg post) ----
hd_break <- NA_character_; orig_ci <- c(NA, NA)
if (have_honest) {
  es_all <- es_ix; rels_all <- rels
  numPre <- sum(rels_all < 0); numPost <- sum(rels_all >= 0)
  bh <- cf[es_all]; Sg <- V[es_all, es_all, drop = FALSE]
  lvec <- rep(1 / numPost, numPost)
  oc <- tryCatch(HonestDiD::constructOriginalCS(bh, Sg, numPre, numPost, l_vec = lvec, alpha = 0.05),
                 error = function(e) NULL)
  if (!is.null(oc)) orig_ci <- c(oc$lb, oc$ub)
  rm <- tryCatch(HonestDiD::createSensitivityResults_relativeMagnitudes(
                   bh, Sg, numPre, numPost, l_vec = lvec,
                   Mbarvec = c(0, 0.5, 1, 1.5, 2), alpha = 0.05),
                 error = function(e) NULL)
  if (!is.null(rm)) {
    inc0 <- (rm$lb <= 0) & (rm$ub >= 0)
    bd <- rm$Mbar[inc0]
    hd_break <- if (length(bd)) sprintf("%.2f", min(bd)) else ">2"
    write.csv(rm, file.path(outdir, sprintf("honestdid_%s.csv", outcome)), row.names = FALSE)
  }
}

## ---- output ----
write.csv(tabmat, file.path(outdir, sprintf("event_study_%s.csv", outcome)), row.names = FALSE)
summ <- list(outcome = outcome, transform = transf, t0 = t0lab, horizon = H,
             matched_treated = n_t, matched_control = n_c, pool_weeks = nrow(d),
             clusters = length(unique(d$cluster_key)),
             cond_trends = condtr, fe = fe_str, filter_class = fclass, filter_tier = ftier,
             joint_leads_p = joint_p, lead_slope_p = slope_p, max_abs_pre = max_abs_pre,
             orig_ci_lo = orig_ci[1], orig_ci_hi = orig_ci[2], honestdid_breakdown_Mbar = hd_break)
write_json(summ, file.path(outdir, sprintf("summary_%s.json", outcome)), auto_unbox = TRUE, digits = 6)

cat("\n=== PRE-TREND (leads, rel<0; rel=-1 ref) ===\n")
cat(sprintf("  %4s %14s %12s %9s %9s\n", "rel", "beta", "se", "wcr_p", "crv1_p"))
for (i in which(tabmat$rel < 0)) cat(sprintf("  %4d %14.4f %12.4f %9.4f %9.4f\n",
    tabmat$rel[i], tabmat$beta[i], tabmat$se[i], tabmat$wcr_p[i], tabmat$crv1_p[i]))
cat("\n=== POST (lags, rel>=0) -- PROVISIONAL ===\n")
for (i in which(tabmat$rel >= 0)) cat(sprintf("  %4d %14.4f %12.4f %9.4f %9.4f\n",
    tabmat$rel[i], tabmat$beta[i], tabmat$se[i], tabmat$wcr_p[i], tabmat$crv1_p[i]))

cat("\n=== SUMMARY ===\n")
cat(sprintf("  outcome/transform      : %s [%s]\n", outcome, transf))
cat(sprintf("  matched treated/control: %d / %d   clusters %d\n", n_t, n_c, summ$clusters))
cat(sprintf("  JOINT leads=0 (Wald) p : %.4f\n", joint_p))
cat(sprintf("  lead linear-slope p    : %.4f\n", slope_p))
cat(sprintf("  max |pre-coef|         : %.4f\n", max_abs_pre))
cat(sprintf("  HonestDiD orig CI (avg): [% .4f, % .4f]\n", orig_ci[1], orig_ci[2]))
cat(sprintf("  HonestDiD breakdown Mbar: %s\n", hd_break))
cat(sprintf("  WCR source             : %s\n", if (have_boot) "fwildclusterboot" else "NOT available (install) / use Rust"))
