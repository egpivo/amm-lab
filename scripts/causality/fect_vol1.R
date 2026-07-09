#!/usr/bin/env Rscript
# Amendment 017: AUXILIARY factor-model counterfactual for token-1 volume (vol1) ONLY.
# NOT a primary causal result, NOT a DiD rescue. Reports against a fixed 5-part gate; any
# signal is described only as "auxiliary trajectory-based evidence on token-1 volume paths."
#
# Estimator: fect (Liu-Wang-Xu) interactive fixed effects (+ matrix completion robustness),
# factor number by CV, bootstrap SE. Matched-overlap units (treated + unique matched controls);
# fect is unit-level so control multiplicity from the DiD does not carry weights here.
#
# Usage: Rscript fect_vol1.R [--prewin 20] [--postwin 8] [--nboots 200]

suppressMessages({ library(fect); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
getopt <- function(k,d){ i<-match(k,args); if(!is.na(i)&&i<length(args)) args[i+1] else d }
ROOT <- Sys.getenv("AMMLAB_ROOT","/Users/joseph/amm-lab")
DATA <- Sys.getenv("AMMLAB_DATA", file.path(ROOT,"data/causality"))
PRE <- as.integer(getopt("--prewin","20")); POST <- as.integer(getopt("--postwin","8"))
NB  <- as.integer(getopt("--nboots","200"))
OUT <- file.path(DATA,"fect_out"); dir.create(OUT, showWarnings=FALSE)
T0 <- "2025-51"

gi <- local({ b0<-as.integer(as.POSIXct("2024-01-01 00:00:00",tz="UTC")); b1<-as.integer(as.POSIXct("2026-06-30 23:59:59",tz="UTC"))
  ts<-seq(b0,b1,by=86400); labs<-sort(unique(format(as.POSIXct(ts,origin="1970-01-01",tz="UTC"),"%Y-%W"))); setNames(seq_along(labs)-1L,labs)})
t0i <- gi[[T0]]

mp <- fromJSON(file.path(DATA,"matched_pairs.json"))
tset <- unique(mp$treated); ctrl <- unique(unlist(mp$controls)); keep <- union(tset,ctrl)
d <- read.csv(file.path(DATA,"panel_weekly_rust.csv"), stringsAsFactors=FALSE,
              colClasses=c(pool="character",unit_role="character",week="character"))
d <- d[d$week %in% names(gi) & d$pool %in% keep,]
d$widx <- gi[d$week]; d$rel <- d$widx - t0i
d <- d[d$rel >= -PRE & d$rel <= POST,]
d$treated <- as.integer(d$pool %in% tset)
d$D <- as.integer(d$treated==1 & d$rel>=0)                 # treatment on for treated post-t0
d$yt <- asinh(d$vol1)
d$uid <- as.integer(factor(d$pool))                        # fect wants integer-ish index ok as factor
cat(sprintf("fect vol1: %d units (%d treated, %d control) | rel [-%d,%d] | %d obs\n",
    length(unique(d$pool)), length(unique(d$pool[d$treated==1])), length(unique(d$pool[d$treated==0])),
    PRE, POST, nrow(d)))

num <- function(x) if(is.null(x)||length(x)==0) NA else as.numeric(x)
J <- list(outcome="vol1", transform="asinh", prewin=PRE, postwin=POST, nboots=NB,
          n_units=length(unique(d$pool)), n_treated=length(unique(d$pool[d$treated==1])),
          n_control=length(unique(d$pool[d$treated==0])), n_obs=nrow(d))

## 1) CV to select factor number (capture the full MSPE path)
set.seed(1)
cvfit <- tryCatch(fect(yt ~ D, data=d, index=c("pool","week"), method="ife", force="two-way",
                       CV=TRUE, r=c(0,6), se=FALSE, parallel=FALSE),
                  error=function(e){cat("CV error:",conditionMessage(e),"\n"); NULL})
if(!is.null(cvfit)){
  J$r_cv <- num(cvfit$r.cv)
  J$cv_mspe <- tryCatch(as.list(setNames(round(cvfit$CV.out[,"MSPE"],5), paste0("r",cvfit$CV.out[,"r"]))),
                        error=function(e) tryCatch(round(cvfit$MSPE,5), error=function(e2) NA))
  cat("CV-selected r*:", J$r_cv, "\n")
}

## 2) canonical fixed r=0 fit with bootstrap SE -> att, CI, pre-fit test, leverage
set.seed(1)
main <- fect(yt ~ D, data=d, index=c("pool","week"), method="ife", force="two-way",
             CV=FALSE, r=0, se=TRUE, nboots=NB, parallel=FALSE)
J$att <- num(main$att.avg)
J$ci  <- tryCatch(as.numeric(main$est.avg[,c("CI.lower","CI.upper")]), error=function(e) c(NA,NA))
J$test_out <- tryCatch(lapply(main$test.out, function(v) round(as.numeric(v),6)), error=function(e) NULL)  # f.stat,f.p,equiv...
eff <- tryCatch(main$eff, error=function(e) NULL)
if(!is.null(eff)){                      # eff is T x N; unit effect = column mean over post periods
  ue <- colMeans(eff, na.rm=TRUE); ue <- ue[is.finite(ue)]
  J$leverage <- list(n=length(ue), mean=round(mean(ue),5), sd=round(sd(ue),5),
                     max_abs_dev_over_sd=round(max(abs(ue-mean(ue)))/sd(ue),3))
}
# plot.fect returns a ggplot object -> save with ggsave (base png()/dev.off() does not capture it)
tryCatch(ggplot2::ggsave(file.path(OUT,"fect_vol1_gap.png"),
                         plot(main, type="gap", main="fect vol1 (r=0): dynamic ATT (auxiliary, gate-failed)"),
                         width=8, height=4.5, dpi=130), error=function(e) cat("plot err:",conditionMessage(e),"\n"))
cat("r=0 att:",J$att," CI:",paste(round(J$ci,4),collapse=","),"\n")

## 3) placebo (fixed r=0)
plc <- tryCatch(fect(yt ~ D, data=d, index=c("pool","week"), method="ife", force="two-way",
                     CV=FALSE, r=0, se=TRUE, nboots=NB, parallel=FALSE, placeboTest=TRUE, placebo.period=c(-3,0)),
                error=function(e){cat("placebo error:",conditionMessage(e),"\n"); NULL})
if(!is.null(plc)){ J$placebo_att <- num(plc$att.avg)
  J$placebo_p <- tryCatch(num(plc$test.out$p.placebo), error=function(e) NA) }

## 4) factor-count sensitivity r=0,1,2
rsens <- list()
for(rr in 0:2){ o <- tryCatch(fect(yt ~ D, data=d, index=c("pool","week"), method="ife", force="two-way",
                     CV=FALSE, r=rr, se=FALSE, parallel=FALSE), error=function(e)NULL)
  rsens[[paste0("r",rr)]] <- if(!is.null(o)) round(num(o$att.avg),5) else NA }
J$factor_sensitivity <- rsens

## 5) matrix-completion robustness
mc <- tryCatch(fect(yt ~ D, data=d, index=c("pool","week"), method="mc", force="two-way",
                    CV=TRUE, se=FALSE, parallel=FALSE), error=function(e)NULL)
if(!is.null(mc)) J$mc_att <- round(num(mc$att.avg),5)

# gate verdict (fixed thresholds, Amendment 017)
J$gate <- list(
  g1_prefit_pass = tryCatch(J$test_out$f.p >= 0.05, error=function(e) NA),
  g2_placebo_flat = tryCatch(abs(J$placebo_att) < abs(J$att) && (is.na(J$placebo_p)||J$placebo_p>=0.05), error=function(e) NA),
  g3_no_leverage = tryCatch(J$leverage$max_abs_dev_over_sd < 5, error=function(e) NA),
  g4_ci_reported_null = tryCatch(J$ci[1] <= 0 && J$ci[2] >= 0, error=function(e) NA),
  g5_factor_stable = tryCatch({v<-unlist(rsens); (max(v)-min(v)) < 0.2}, error=function(e) NA))
J$gate_pass_all <- all(unlist(J$gate)==TRUE, na.rm=FALSE)

write_json(J, file.path(OUT,"fect_vol1.json"), auto_unbox=TRUE, digits=6, pretty=TRUE, na="null")
cat("\n=== GATE ===\n"); print(J$gate); cat("gate_pass_all:", J$gate_pass_all, "\n")
cat("wrote fect_out/fect_vol1.json (+ gap png)\n")
