#!/usr/bin/env Rscript
# Amendment 015, step 1: FPCA diagnostic on the PRE-PERIOD trajectories of a flow/revenue
# outcome that fails the primary parallel-trends gate (vol1, lp_fee_income). Question:
# is the failure consistent with heterogeneous latent-factor loadings (treated vs matched
# control load differently on common market-flow factors), rather than noise? NO post-period
# ATT is estimated here.
#
# Outputs (per outcome, into fpca_out/):
#   - mean pre-period trajectory, treated vs matched control
#   - first two FPC eigenfunctions (the common flow/revenue factors)
#   - FPC-score scatter (xi1 vs xi2) treated vs control + a permutation test of score imbalance
#   - a small JSON summary (explained variance, imbalance p-value)
#
# Usage: Rscript fpca_diagnostic.R --outcome vol1 [--t0 2025-51] [--prewin 12] [--transform asinh]

suppressMessages({ library(fdapace); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(k, d) { i <- match(k, args); if (!is.na(i) && i < length(args)) args[i+1] else d }
ROOT <- Sys.getenv("AMMLAB_ROOT", "/Users/joseph/amm-lab")
DATA <- Sys.getenv("AMMLAB_DATA", file.path(ROOT, "data/causality"))
outcome <- getopt("--outcome", "vol1")
t0lab   <- getopt("--t0", "2025-51")
prewin  <- as.integer(getopt("--prewin", "12"))   # use pre-period weeks rel in [-prewin, -1]
transf  <- getopt("--transform", "asinh")
outdir  <- getopt("--out", file.path(DATA, "fpca_out")); dir.create(outdir, showWarnings = FALSE)

grid_index <- function() {
  b0 <- as.integer(as.POSIXct("2024-01-01 00:00:00", tz = "UTC"))
  b1 <- as.integer(as.POSIXct("2026-06-30 23:59:59", tz = "UTC"))
  ts <- seq(b0, b1, by = 86400)
  setNames(seq_along(sort(unique(format(as.POSIXct(ts, origin = "1970-01-01", tz = "UTC"), "%Y-%W")))) - 1L,
           sort(unique(format(as.POSIXct(ts, origin = "1970-01-01", tz = "UTC"), "%Y-%W"))))
}
gi <- grid_index(); t0i <- gi[[t0lab]]

mp <- fromJSON(file.path(DATA, "matched_pairs.json"))
treated_set <- unique(mp$treated); ctrl_set <- unique(unlist(mp$controls))
keep <- union(treated_set, ctrl_set)

d <- read.csv(file.path(DATA, "panel_weekly_rust.csv"), stringsAsFactors = FALSE,
              colClasses = c(pool = "character", unit_role = "character", week = "character"))
d <- d[d$week %in% names(gi) & d$pool %in% keep, ]
d$rel <- gi[d$week] - t0i
d <- d[d$rel >= -prewin & d$rel <= -1, ]                      # PRE-PERIOD ONLY (no post)
d$treated <- as.integer(d$pool %in% treated_set)
d$y <- switch(transf, asinh = asinh(d[[outcome]]), log1p = log1p(pmax(d[[outcome]], 0)), d[[outcome]])

# sparse functional data: one trajectory per pool over pre-period rel.
# CENTER each pool at its own pre-period mean: pool level is absorbed by pool FE in the DiD
# and is not what breaks parallel trends; centering makes the FPCs capture trend/SHAPE, so the
# treated-vs-control score imbalance tests DIFFERENTIAL TRENDS (the actual PT-failure mode).
pools <- unique(d$pool)
Ly <- lapply(pools, function(p) { v <- d$y[d$pool == p][order(d$rel[d$pool == p])]; v - mean(v) })
Lt <- lapply(pools, function(p) sort(d$rel[d$pool == p]))
trt <- as.integer(pools %in% treated_set)
ok <- vapply(Ly, function(v) length(v) >= 3, logical(1))      # need >=3 pre points for FPCA
Ly <- Ly[ok]; Lt <- Lt[ok]; trt <- trt[ok]; pools <- pools[ok]
cat(sprintf("FPCA pre-period [%s, rel -%d..-1]: %d pools (%d treated, %d control)\n",
            outcome, prewin, length(pools), sum(trt), sum(!trt)))

fp <- FPCA(Ly, Lt, optns = list(dataType = "Sparse", nRegGrid = 51, methodSelectK = "FVE", FVEthreshold = 0.95))
K <- min(3, ncol(fp$xiEst)); xi <- fp$xiEst[, 1:K, drop = FALSE]
fve <- cumsum(fp$lambda) / sum(fp$lambda)
cat(sprintf("  FPCs: K=%d, FVE(1..K)=%s\n", K, paste(sprintf("%.2f", fve[1:K]), collapse = ",")))

# permutation test of treated/control separation in FPC-score space (Mahalanobis of mean diff)
S <- cov(xi); Si <- MASS::ginv(S)
md <- function(g) { m1 <- colMeans(xi[g == 1, , drop = FALSE]); m0 <- colMeans(xi[g == 0, , drop = FALSE])
                    as.numeric(t(m1 - m0) %*% Si %*% (m1 - m0)) }
obs <- md(trt); set.seed(1)
perm <- replicate(2000, md(sample(trt)))
pval <- mean(perm >= obs)
cat(sprintf("  score imbalance (Mahalanobis mean-diff): obs=%.3f perm-p=%.4f\n", obs, pval))

# ---- plots ----
png(file.path(outdir, sprintf("fpca_%s.png", outcome)), width = 1500, height = 500, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
# (1) mean pre-period trajectory treated vs control on the FPCA reg grid
tg <- fp$workGrid
mu_all <- fp$mu
fit <- fitted(fp)                        # smoothed trajectories per pool on workGrid
mt <- colMeans(fit[trt == 1, , drop = FALSE]); mc <- colMeans(fit[trt == 0, , drop = FALSE])
matplot(tg, cbind(mt, mc), type = "l", lty = 1, lwd = 2, col = c("#8E3B3B", "#4C6A91"),
        xlab = "event time (pre)", ylab = sprintf("mean %s [%s]", outcome, transf),
        main = "Pre-period mean trajectory")
legend("topleft", c("treated", "control"), col = c("#8E3B3B", "#4C6A91"), lwd = 2, bty = "n")
# (2) first two eigenfunctions
matplot(tg, fp$phi[, 1:min(2, K), drop = FALSE], type = "l", lty = 1, lwd = 2,
        col = c("#1F1F1F", "#B7905E"), xlab = "event time (pre)", ylab = "phi_k(t)",
        main = "FPC eigenfunctions")
legend("topleft", paste0("FPC", 1:min(2, K)), col = c("#1F1F1F", "#B7905E"), lwd = 2, bty = "n")
# (3) score scatter
plot(xi[, 1], xi[, min(2, K)], col = ifelse(trt == 1, "#8E3B3B", "#4C6A91"), pch = 19, cex = 0.5,
     xlab = "FPC1 score", ylab = "FPC2 score",
     main = sprintf("Score imbalance (perm p=%.3f)", pval))
legend("topleft", c("treated", "control"), col = c("#8E3B3B", "#4C6A91"), pch = 19, bty = "n")
dev.off()

write_json(list(outcome = outcome, transform = transf, prewin = prewin,
                n_pools = length(pools), n_treated = sum(trt), n_control = sum(!trt),
                K = K, fve = round(fve[1:K], 4),
                score_imbalance_maha = round(obs, 4), score_imbalance_permp = round(pval, 4)),
           file.path(outdir, sprintf("fpca_%s.json", outcome)), auto_unbox = TRUE)
cat(sprintf("wrote %s/fpca_%s.{png,json}\n", outdir, outcome))
