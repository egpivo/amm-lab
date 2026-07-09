#!/usr/bin/env Rscript
# Robustness battery for the IDENTIFIED (pass-parallel-trends) primary outcomes, ±8 window.
# Locks the primary null with: (a) aggregate DiD ATT + Cinelli-Hazlett robustness value
# (sensemakr), (b) entropy-balancing balance check + reweighted ATT (ebal). Placebo dates and
# caliper sensitivity are run separately. NOTHING here is promoted to the manuscript pre-review.
#
# Usage: Rscript robustness_battery.R  (loops the primary outcomes)

suppressMessages({ library(fixest); library(jsonlite); library(sensemakr); library(ebal) })
ROOT <- Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA <- Sys.getenv("AMMLAB_DATA", file.path(ROOT, "data/causality"))
T0 <- "2025-51"; H <- 8
OUTS <- c("twl_active_liquidity", "depth_1pct", "depth_2pct", "depth_5pct")

gi <- local({
  b0 <- as.integer(as.POSIXct("2024-01-01 00:00:00", tz = "UTC"))
  b1 <- as.integer(as.POSIXct("2026-06-30 23:59:59", tz = "UTC"))
  ts <- seq(b0, b1, by = 86400)
  labs <- sort(unique(format(as.POSIXct(ts, origin = "1970-01-01", tz = "UTC"), "%Y-%W")))
  setNames(seq_along(labs) - 1L, labs)
}); t0i <- gi[[T0]]

mp <- fromJSON(file.path(DATA, "matched_pairs.json"))
tset <- unique(mp$treated); cc <- table(unlist(mp$controls))
wof <- c(setNames(rep(1, length(tset)), tset), setNames(as.numeric(cc), names(cc)))
tok <- fromJSON(file.path(DATA, "ckpt_tokens.json"))
cof <- vapply(tok, function(p){ s <- sort(tolower(p[nzchar(p)])); if(length(s)) paste(s, collapse="-") else "na" }, character(1))
fr <- read.csv(file.path(DATA, "feerev_panelvars.csv"), stringsAsFactors = FALSE, colClasses = c(pool="character"))
S_of <- setNames(suppressWarnings(as.numeric(fr$fr12_usd)), fr$pool)
tier_of <- setNames(as.character(fr$tier), fr$pool); class_of <- setNames(as.character(fr$class), fr$pool)

panel <- read.csv(file.path(DATA, "panel_weekly_rust.csv"), stringsAsFactors = FALSE,
                  colClasses = c(pool="character", unit_role="character", week="character"))
panel <- panel[panel$week %in% names(gi) & panel$pool %in% names(wof), ]
panel$rel <- gi[panel$week] - t0i

smd <- function(x, g, w) {  # standardized mean diff treated vs weighted control
  mt <- mean(x[g==1]); mc <- weighted.mean(x[g==0], w[g==0]); s <- sd(x)
  if (s == 0) 0 else (mt - mc) / s
}

res <- list()
for (o in OUTS) {
  d <- panel[abs(panel$rel) <= H & is.finite(panel[[o]]), ]
  d$yt <- asinh(d[[o]]); d$treated <- as.integer(d$pool %in% tset)
  d$w <- wof[d$pool]; d$cluster_key <- factor(ifelse(!is.na(cof[d$pool]), cof[d$pool], "na"))
  d$post <- as.integer(d$rel >= 0)
  d$pool <- factor(d$pool); d$week <- factor(d$week)
  # purge singletons for a clean fit
  repeat { m0 <- feols(yt ~ treated:post | pool + week, d, weights=~w, cluster=~cluster_key)
           u <- fixest::obs(m0); if (length(u)==nrow(d)) break; d <- droplevels(d[u,,drop=FALSE]) }
  ct <- coeftable(m0); att <- ct["treated:post","Estimate"]; se <- ct["treated:post","Std. Error"]
  tstat <- att/se; dof <- fixest::degrees_freedom(m0, type="t")
  rv <- tryCatch(sensemakr::robustness_value(t_statistic = tstat, dof = dof, q = 1)[[1]],
                 error = function(e) NA_real_)

  # ---- entropy balancing on pre-period covariates (pool level) ----
  pre <- d[d$post==0, ]
  pm <- tapply(pre$yt, pre$pool, mean)
  ps <- tapply(seq_len(nrow(pre)), pre$pool, function(ix){
          v <- pre$yt[ix]; r <- pre$rel[ix]
          if (length(v) > 1 && sd(r) > 0) unname(coef(lm(v ~ r))[2]) else 0 })
  pu <- unique(as.character(d$pool)); trt <- as.integer(pu %in% tset)
  X <- cbind(logS = log1p(pmax(S_of[pu], 0)), premean = pm[pu], preslope = ps[pu])
  X[!is.finite(X)] <- 0
  wt_before <- wof[pu]
  eb <- tryCatch(ebal::ebalance(Treatment = trt, X = X, print.level = -1), error = function(e) NULL)
  wt_after <- wt_before
  if (!is.null(eb)) { wt_after[trt==0] <- eb$w; wt_after[trt==1] <- 1 }
  smd_before <- sapply(1:ncol(X), function(j) smd(X[,j], trt, wt_before))
  smd_after  <- sapply(1:ncol(X), function(j) smd(X[,j], trt, wt_after))

  # reweighted ATT: map pool ebal weight onto rows (times nothing; ebal replaces freq weight for controls)
  ew <- wt_after[as.character(d$pool)]
  m1 <- feols(yt ~ treated:post | pool + week, d, weights = ew, cluster = ~cluster_key)
  c1 <- coeftable(m1)["treated:post", ]

  res[[o]] <- list(outcome=o, att=att, se=se, ci=att+c(-1,1)*1.96*se, rv=rv, dof=dof,
                   smd_logS=c(smd_before[1],smd_after[1]),
                   att_ebal=c1["Estimate"], se_ebal=c1["Std. Error"],
                   ci_ebal=c1["Estimate"]+c(-1,1)*1.96*c1["Std. Error"])
  cat(sprintf("%-24s ATT %+.3f [% .3f,% .3f]  RV=%.3f  | ebalATT %+.3f [% .3f,% .3f]  SMD(logS) %.2f->%.2f\n",
      o, att, att-1.96*se, att+1.96*se, rv, c1["Estimate"], c1["Estimate"]-1.96*c1["Std. Error"],
      c1["Estimate"]+1.96*c1["Std. Error"], smd_before[1], smd_after[1]))
}
write_json(res, file.path(DATA, "analysis_r_out", "robustness_battery.json"), auto_unbox=TRUE, digits=6)
cat("wrote robustness_battery.json\n")
