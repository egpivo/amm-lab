#!/usr/bin/env Rscript
# HonestDiD (Rambachan-Roth 2023) sensitivity on the event-study coefficients exported by
# estimate_event_study.py. Answers the marginal pre-trend slope: how large a violation of
# parallel trends (relative to the observed pre-period) would overturn the post-period
# conclusion. Reports the original CI + relative-magnitude robust CIs over an Mbar grid.
#
# Usage: Rscript honest_did.R <es_beta_vcov_*.json> [out_csv]

suppressMessages({ library(jsonlite); library(HonestDiD) })

args <- commandArgs(trailingOnly = TRUE)
inp <- if (length(args) >= 1) args[1] else
  "/Users/joseph/amm-lab/.local/amm_paper_c/data/event_study_py_out/es_beta_vcov_twl_active_liquidity.json"
outcsv <- if (length(args) >= 2) args[2] else sub("\\.json$", "_honestdid.csv", inp)

j <- fromJSON(inp)
betahat <- as.numeric(j$beta)
sigma <- matrix(unlist(j$vcov), nrow = length(betahat), byrow = TRUE)
numPre <- as.integer(j$num_pre)
numPost <- as.integer(j$num_post)
cat(sprintf("outcome %s [%s] | pre %d post %d | rel [%d..%d]\n",
            j$outcome, j$transform, numPre, numPost, min(j$rel), max(j$rel)))

# target = average post-period effect (aggregate ATT on the transformed scale)
l_vec <- rep(1 / numPost, numPost)

orig <- constructOriginalCS(betahat = betahat, sigma = sigma,
                            numPrePeriods = numPre, numPostPeriods = numPost,
                            l_vec = l_vec, alpha = 0.05)
cat(sprintf("\nOriginal 95%% CI (avg post, asinh scale): [% .4f, % .4f]\n", orig$lb, orig$ub))

Mbarvec <- c(0, 0.5, 1, 1.5, 2)
rm <- createSensitivityResults_relativeMagnitudes(
  betahat = betahat, sigma = sigma,
  numPrePeriods = numPre, numPostPeriods = numPost,
  l_vec = l_vec, Mbarvec = Mbarvec, alpha = 0.05)

rm$includes_zero <- (rm$lb <= 0) & (rm$ub >= 0)
cat("\nRelative-magnitude robust 95% CIs (avg post):\n")
cat(sprintf("  %-6s %10s %10s %s\n", "Mbar", "lb", "ub", "incl 0?"))
for (i in seq_len(nrow(rm))) {
  cat(sprintf("  %-6.2f % 10.4f % 10.4f   %s\n",
              rm$Mbar[i], rm$lb[i], rm$ub[i], ifelse(rm$includes_zero[i], "yes", "NO")))
}

# breakdown Mbar: smallest Mbar at which the robust CI first includes 0
bd <- rm$Mbar[which(rm$includes_zero)]
cat(sprintf("\nbreakdown Mbar (robust CI first includes 0): %s\n",
            if (length(bd)) sprintf("%.2f", min(bd)) else "not within grid (still excludes 0 at Mbar=2)"))

write.csv(rm[, c("Mbar", "lb", "ub", "includes_zero")], outcsv, row.names = FALSE)
cat(sprintf("wrote %s\n", outcsv))
