//! Event-study design: treated $\times \mathbf{1}[t - t_{0i} = k]$ indicators with the
//! reference bin $k = -1$ omitted. Controls contribute zeros (they identify the common
//! calendar path through the week fixed effect).

use std::collections::BTreeSet;

/// Sorted unique event-time bins among treated observations, excluding the reference `-1`.
pub fn bins_from(event_time: &[i64], treated: &[bool]) -> Vec<i64> {
    let mut s: BTreeSet<i64> = BTreeSet::new();
    for (&et, &tr) in event_time.iter().zip(treated) {
        if tr && et != -1 {
            s.insert(et);
        }
    }
    s.into_iter().collect()
}

/// One column per bin: `treated * 1[event_time == k]`.
pub fn event_time_columns(event_time: &[i64], treated: &[bool], bins: &[i64]) -> Vec<Vec<f64>> {
    bins.iter()
        .map(|&k| {
            event_time
                .iter()
                .zip(treated)
                .map(|(&et, &tr)| if tr && et == k { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omits_reference_bin_and_sorts() {
        let et = vec![-2, -1, 0, 1, -1, 0];
        let tr = vec![true, true, true, true, false, false];
        assert_eq!(bins_from(&et, &tr), vec![-2, 0, 1]);
    }

    #[test]
    fn indicators_zero_for_controls() {
        let et = vec![0, 0];
        let tr = vec![true, false];
        let cols = event_time_columns(&et, &tr, &[0]);
        assert_eq!(cols[0], vec![1.0, 0.0]);
    }
}
