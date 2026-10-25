//! Token timestamps from the alignment heads' cross-attention, as transformers' _extract_token_timestamps:
//! crop to the audio frames, normalise each head over tokens, median filter (width 7) over time, average the
//! heads, then dynamic time warping. fp16 roundings follow the reference, which runs this on fp16 tensors.
use crate::gpu::rh;

/// w: [rows][heads][cols_full] probabilities (rows = generated tokens fed back as input); returns the time
/// (seconds) at which each row starts, one per row.
pub fn jump_times(w: &[f32], rows: usize, heads: usize, cols_full: usize, cols: usize, width: usize) -> Vec<f32> {
    if cols == 0 {
        return vec![0.0; rows];
    }
    let at = |i: usize, a: usize, j: usize| w[(i * heads + a) * cols_full + j];
    let mut avg = vec![0f32; rows * cols];
    let pad = width / 2;
    let mut z = vec![0f32; rows * cols];
    for a in 0..heads {
        for j in 0..cols {
            let mean_f = (0..rows).map(|i| at(i, a, j)).sum::<f32>() / rows as f32;
            let var = (0..rows).map(|i| (at(i, a, j) - mean_f).powi(2)).sum::<f32>() / rows as f32;
            let (mean, std) = (rh(mean_f), rh(var.sqrt()));
            for i in 0..rows {
                // a constant column (always so for a single row) scores 0 here; the reference divides by zero (NaN)
                z[i * cols + j] = if std > 0.0 { rh(rh(at(i, a, j) - mean) / std) } else { 0.0 };
            }
        }
        for i in 0..rows {
            let row = &z[i * cols..(i + 1) * cols];
            for j in 0..cols {
                let v = if cols <= pad {
                    row[j]
                } else {
                    let mut win: Vec<f32> = (0..width)
                        .map(|k| {
                            let t = j as isize + k as isize - pad as isize;
                            let t = if t < 0 {
                                -t
                            } else if t >= cols as isize {
                                2 * (cols as isize - 1) - t
                            } else {
                                t
                            };
                            row[t as usize]
                        })
                        .collect();
                    win.sort_by(|x, y| x.total_cmp(y));
                    win[pad]
                };
                avg[i * cols + j] += v;
            }
        }
    }
    let m: Vec<f64> = avg.iter().map(|&s| -(rh(s / heads as f32) as f64)).collect();
    let (text, time) = dtw(&m, rows, cols);
    let mut out = Vec::with_capacity(rows);
    for k in 0..text.len() {
        if k == 0 || text[k] != text[k - 1] {
            out.push((time[k] as f64 * 0.02) as f32);
        }
    }
    // one time per row, also when degenerate scores (f16 overflow) bend the path
    out.resize(rows, out.last().copied().unwrap_or(0.0));
    out
}

/// numpy reference semantics: float32 cost table, ties broken towards the horizontal move.
fn dtw(m: &[f64], rows: usize, cols: usize) -> (Vec<isize>, Vec<isize>) {
    let w = cols + 1;
    let mut cost = vec![f32::INFINITY; (rows + 1) * w];
    let mut trace = vec![-1i8; (rows + 1) * w];
    cost[0] = 0.0;
    for j in 1..=cols {
        for i in 1..=rows {
            let (c0, c1, c2) = (cost[(i - 1) * w + j - 1], cost[(i - 1) * w + j], cost[i * w + j - 1]);
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };
            cost[i * w + j] = (m[(i - 1) * cols + j - 1] + c as f64) as f32;
            trace[i * w + j] = t;
        }
    }
    trace[..w].fill(2);
    for i in 0..=rows {
        trace[i * w] = 1;
    }
    let (mut i, mut j) = (rows, cols);
    let (mut text, mut time) = (vec![], vec![]);
    while i > 0 || j > 0 {
        text.push(i as isize - 1);
        time.push(j as isize - 1);
        match trace[i * w + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    text.reverse();
    time.reverse();
    (text, time)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_follows_the_cheap_diagonal() {
        // 3 tokens x 6 frames: token i <-> frames 2i, 2i + 1 cost -1, every other cell +1
        let (rows, cols) = (3, 6);
        let m: Vec<f64> = (0..rows * cols).map(|k| if (k % cols) / 2 == k / cols { -1.0 } else { 1.0 }).collect();
        let (text, time) = dtw(&m, rows, cols);
        assert_eq!(text, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(time, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn jump_times_of_a_clean_alignment() {
        // one head; token i attends to frames 10i..10i+9 of 40 audio frames (out of 60 in the buffer)
        let (rows, cols_full, cols) = (4, 60, 40);
        let mut w = vec![0f32; rows * cols_full];
        for i in 0..rows {
            for j in 10 * i..10 * i + 10 {
                w[i * cols_full + j] = 0.1;
            }
        }
        let t = jump_times(&w, rows, 1, cols_full, cols, 7);
        assert_eq!(t.len(), rows);
        assert_eq!(jump_times(&w, rows, 1, cols_full, 0, 7), vec![0.0; rows]); // no audio frames
        assert_eq!(jump_times(&w[..cols_full], 1, 1, cols_full, cols, 7), vec![0.0]); // one token
        for (i, &ti) in t.iter().enumerate() {
            assert!((ti - 0.2 * i as f32).abs() < 0.03, "token {i} at {ti}");
        }
    }
}
