//! WAV reading, resampling to 16 kHz and Whisper's log-mel features (WhisperFeatureExtractor, torch path):
//! zero-pad to 30 s, centred 400-point STFT with a periodic Hann window and reflect padding, power spectrum,
//! 80 or 128 Slaney mel filters up to 8 kHz, log10 floored at 1e-10, clamped to (max - 8), then (x + 4) / 4.
use crate::gpu::R;

pub const SR: usize = 16000;
pub const N_SAMPLES: usize = 30 * SR;
pub const N_FRAMES: usize = 3000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = N_FFT / 2 + 1;

/// Mono samples in [-1, 1) and the sample rate. Channels are averaged (librosa's mono=True).
pub fn read_wav(path: &str) -> R<(Vec<f32>, usize)> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("{path}: not a RIFF/WAVE file").into());
    }
    let u16le = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32le = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let (mut fmt, mut ch, mut sr, mut bits) = (0u16, 0usize, 0usize, 0u16);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32le(i + 4) as usize;
        let body = i + 8;
        if body + n > b.len() && id != b"data" {
            return Err(format!("{path}: truncated {} chunk", String::from_utf8_lossy(id)).into());
        }
        if id == b"fmt " {
            if n < 16 || (u16le(body) == 0xfffe && n < 26) {
                return Err(format!("{path}: short fmt chunk").into());
            }
            fmt = u16le(body);
            ch = u16le(body + 2) as usize;
            sr = u32le(body + 4) as usize;
            bits = u16le(body + 14);
            if fmt == 0xfffe {
                fmt = u16le(body + 24); // WAVE_FORMAT_EXTENSIBLE: sub-format GUID starts with the format code
            }
        } else if id == b"data" {
            if ch == 0 || bits == 0 {
                return Err(format!("{path}: data before fmt").into());
            }
            if !matches!((fmt, bits), (1, 16) | (1, 24) | (1, 32) | (3, 32)) {
                return Err(format!("{path}: unsupported WAV format {fmt}/{bits} bit (16/24/32-bit PCM or 32-bit float)").into());
            }
            let frame = ch * bits as usize / 8;
            // a streaming writer leaves the size at 0xFFFFFFFF: then the data runs to the end of the file
            let streamed = n == 0xffff_ffff;
            if !streamed && (body + n > b.len() || !n.is_multiple_of(frame)) {
                return Err(format!("{path}: data chunk is truncated or not whole frames").into());
            }
            let data = if streamed { &b[body..] } else { &b[body..body + n] };
            let nf = data.len() / frame;
            let sample = |k: usize| -> f32 {
                let o = k * bits as usize / 8;
                match (fmt, bits) {
                    (1, 16) => i16::from_le_bytes([data[o], data[o + 1]]) as f32 / 32768.0,
                    (1, 24) => (i32::from_le_bytes([0, data[o], data[o + 1], data[o + 2]]) >> 8) as f32 / 8_388_608.0,
                    (1, 32) => i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0,
                    _ => f32::from_le_bytes(data[o..o + 4].try_into().unwrap()),
                }
            };
            let y: Vec<f32> = (0..nf).map(|t| (0..ch).map(|c| sample(t * ch + c)).sum::<f32>() / ch as f32).collect();
            if y.iter().any(|v| !v.is_finite()) {
                return Err(format!("{path}: samples that are not finite numbers").into());
            }
            return Ok((y, sr));
        }
        i = body + n + (n & 1);
    }
    Err(format!("{path}: no data chunk").into())
}

fn bessel_i0(x: f64) -> f64 {
    let (mut s, mut t, mut k) = (1.0, 1.0, 1.0);
    while t > 1e-12 * s {
        t *= (x / (2.0 * k)).powi(2);
        s += t;
        k += 1.0;
    }
    s
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Resampling to 16 kHz with a linear-phase Kaiser-windowed sinc low-pass (passband to 95% of the lower Nyquist
/// frequency, 32 zero crossings per side, beta 9), evaluated per output phase: an integer ratio has one phase,
/// 44.1 kHz has 160. librosa uses soxr; the difference stays in the top mel bins.
pub fn to_16k(y: &[f32], sr: usize) -> R<Vec<f32>> {
    if sr == SR {
        return Ok(y.to_vec());
    }
    if !(1000..=768_000).contains(&sr) {
        return Err(format!("unsupported sample rate {sr}").into());
    }
    let g = gcd(sr, SR);
    let (up, down) = (SR / g, sr / g); // output k sits at input position k * down / up
    let scale = (SR as f64 / sr as f64).min(1.0); // cutoff relative to the input rate
    let fc = 0.95 * scale / 2.0; // cycles per input sample
    let half = (32.0 / scale).ceil() as isize;
    let beta = 9.0;
    let i0b = bessel_i0(beta);
    let taps = |ph: usize| -> Vec<f64> {
        let frac = ph as f64 / up as f64;
        let taps: Vec<f64> = (-half..=half + 1)
            .map(|n| {
                let t = n as f64 - frac;
                let sinc = if t == 0.0 { 2.0 * fc } else { (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t) };
                let r = t / half as f64;
                if r.abs() > 1.0 { 0.0 } else { sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b }
            })
            .collect();
        let norm: f64 = taps.iter().sum();
        taps.into_iter().map(|w| w / norm).collect()
    };
    // the filters of every phase are tabulated unless the ratio is awkward enough to make that table large
    let table: Option<Vec<Vec<f64>>> = (up * (2 * half as usize + 2) <= 1 << 22).then(|| (0..up).map(taps).collect());
    let n_out = (y.len() * up).div_ceil(down);
    Ok((0..n_out)
        .map(|k| {
            let pos = k * down;
            let (c, ph) = ((pos / up) as isize, pos % up);
            let computed;
            let w = match &table {
                Some(t) => &t[ph],
                None => {
                    computed = taps(ph);
                    &computed
                }
            };
            let mut acc = 0.0f64;
            for (n, &w) in w.iter().enumerate() {
                let idx = c + n as isize - half;
                if idx >= 0 && (idx as usize) < y.len() {
                    acc += w * y[idx as usize] as f64;
                }
            }
            acc as f32
        })
        .collect())
}

fn hz_to_mel(f: f64) -> f64 {
    let logstep = 27.0 / 6.4f64.ln();
    if f >= 1000.0 { 15.0 + (f / 1000.0).ln() * logstep } else { 3.0 * f / 200.0 }
}

fn mel_to_hz(m: f64) -> f64 {
    let logstep = 6.4f64.ln() / 27.0;
    if m >= 15.0 { 1000.0 * (logstep * (m - 15.0)).exp() } else { 200.0 * m / 3.0 }
}

/// Precomputed tables for the features: mel filters [N_BINS][n_mels] and windowed DFT bases.
pub struct Mel {
    n_mels: usize,
    filters: Vec<f32>,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Mel {
    pub fn new(n_mels: usize) -> Self {
        let fft_freqs: Vec<f64> = (0..N_BINS).map(|i| i as f64 * (SR as f64 / 2.0) / (N_BINS - 1) as f64).collect();
        let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let ff: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
        let mut filters = vec![0f32; N_BINS * n_mels];
        for (k, &fq) in fft_freqs.iter().enumerate() {
            for m in 0..n_mels {
                let down = (fq - ff[m]) / (ff[m + 1] - ff[m]);
                let up = (ff[m + 2] - fq) / (ff[m + 2] - ff[m + 1]);
                let enorm = 2.0 / (ff[m + 2] - ff[m]);
                filters[k * n_mels + m] = (down.min(up).max(0.0) * enorm) as f32;
            }
        }
        let win: Vec<f64> = (0..N_FFT).map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos()).collect();
        let mut cos = vec![0f64; N_BINS * N_FFT];
        let mut sin = vec![0f64; N_BINS * N_FFT];
        for k in 0..N_BINS {
            for n in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                cos[k * N_FFT + n] = win[n] * a.cos();
                sin[k * N_FFT + n] = win[n] * a.sin();
            }
        }
        Mel { n_mels, filters, cos, sin }
    }

    /// Log-mel features [n_mels][N_FRAMES] of the first 30 s, and the number of frames that hold audio
    /// (the feature extractor's attention mask sum, used to crop the cross-attention for timestamps).
    pub fn features(&self, y: &[f32]) -> (Vec<f32>, usize) {
        let n_mels = self.n_mels;
        let n = y.len().min(N_SAMPLES);
        let at = |i: isize| -> f64 {
            // the 30 s buffer (zero padded) with centre reflect padding of N_FFT / 2
            let i = if i < 0 {
                -i
            } else if i >= N_SAMPLES as isize {
                2 * (N_SAMPLES as isize - 1) - i
            } else {
                i
            };
            let i = i as usize;
            if i < n { y[i] as f64 } else { 0.0 }
        };
        // frames past the audio are all zero: power 0, which the log floor turns into -10
        let live = ((n + N_FFT / 2).div_ceil(HOP)).min(N_FRAMES);
        let mut logmel = vec![-10f32; n_mels * N_FRAMES];
        let mut frame = [0f64; N_FFT];
        let mut power = [0f32; N_BINS];
        for t in 0..live {
            for (j, v) in frame.iter_mut().enumerate() {
                *v = at((t * HOP + j) as isize - (N_FFT / 2) as isize);
            }
            for (k, pw) in power.iter_mut().enumerate() {
                let (c, s) = (&self.cos[k * N_FFT..(k + 1) * N_FFT], &self.sin[k * N_FFT..(k + 1) * N_FFT]);
                let (mut re, mut im) = (0f64, 0f64);
                for ((&x, &cj), &sj) in frame.iter().zip(c).zip(s) {
                    re += x * cj;
                    im += x * sj;
                }
                *pw = (re * re + im * im) as f32;
            }
            for m in 0..n_mels {
                let acc = power.iter().enumerate().fold(0f32, |a, (k, &p)| a + self.filters[k * n_mels + m] * p);
                logmel[m * N_FRAMES + t] = acc.max(1e-10).log10();
            }
        }
        let mx = logmel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in logmel.iter_mut() {
            *v = (v.max(mx - 8.0) + 4.0) / 4.0;
        }
        (logmel, n.div_ceil(HOP).min(N_FRAMES))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(fmt: u16, ch: u16, sr: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&fmt.to_le_bytes());
        b.extend_from_slice(&ch.to_le_bytes());
        b.extend_from_slice(&sr.to_le_bytes());
        b.extend_from_slice(&(sr * ch as u32 * bits as u32 / 8).to_le_bytes());
        b.extend_from_slice(&(ch * bits / 8).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    fn read(bytes: &[u8]) -> R<(Vec<f32>, usize)> {
        let p = std::env::temp_dir().join(format!("we-wav-{}-{}.wav", std::process::id(), bytes.len()));
        std::fs::write(&p, bytes).unwrap();
        let r = read_wav(p.to_str().unwrap());
        std::fs::remove_file(&p).unwrap();
        r
    }

    #[test]
    fn wav_formats() {
        let pcm16: Vec<u8> = [16384i16, -32768, 0, 8192].iter().flat_map(|v| v.to_le_bytes()).collect();
        let (y, sr) = read(&wav(1, 2, 48000, 16, &pcm16)).unwrap();
        assert_eq!(sr, 48000);
        assert_eq!(y, vec![-0.25, 0.125]); // stereo frames averaged
        let f32le: Vec<u8> = [0.5f32, -1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(read(&wav(3, 1, 16000, 32, &f32le)).unwrap().0, vec![0.5, -1.0]);
        let pcm24 = [0x00, 0x00, 0x40, 0x00, 0x00, 0xc0];
        assert_eq!(read(&wav(1, 1, 16000, 24, &pcm24)).unwrap().0, vec![0.5, -0.5]);
    }

    #[test]
    fn wav_malformed_is_an_error() {
        assert!(read(b"RIFF\0\0\0\0WAVE").is_err()); // no chunks
        assert!(read(&wav(1, 1, 16000, 8, &[0, 1])).is_err()); // 8-bit is not supported
        assert!(read(&wav(1, 1, 16000, 16, &[]).as_slice()[..30]).is_err()); // fmt chunk cut short
        assert!(read(b"not a wav file").is_err());
        let mut cut = wav(1, 1, 16000, 16, &[0, 1, 2, 3]);
        cut.truncate(cut.len() - 1);
        assert!(read(&cut).is_err()); // data chunk shorter than its size
        assert!(read(&wav(1, 1, 16000, 16, &[0, 1, 2])).is_err()); // half a frame
        let mut streamed = wav(1, 1, 16000, 16, &[0, 0x40, 0, 0x40]);
        streamed[40..44].copy_from_slice(&0xffff_ffffu32.to_le_bytes());
        assert_eq!(read(&streamed).unwrap().0, vec![0.5, 0.5]); // size left unset by a streaming writer
    }

    fn tone(sr: usize, hz: f64, secs: f64) -> Vec<f32> {
        (0..(sr as f64 * secs) as usize).map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / sr as f64).sin() as f32).collect()
    }

    fn rms(y: &[f32]) -> f64 {
        let m = &y[y.len() / 4..3 * y.len() / 4]; // away from the edges
        (m.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / m.len() as f64).sqrt()
    }

    #[test]
    fn resampler_keeps_the_band_and_removes_aliases() {
        for sr in [8000, 22050, 44100, 48000] {
            let y = to_16k(&vec![1.0; sr], sr).unwrap();
            assert_eq!(y.len(), 16000);
            assert!((y[8000] - 1.0).abs() < 1e-4, "{sr}: DC gain {}", y[8000]);
            let pass = rms(&to_16k(&tone(sr, 1000.0, 1.0), sr).unwrap());
            assert!((pass - 0.5f64.sqrt()).abs() < 0.01, "{sr}: 1 kHz rms {pass}");
            if sr > 16000 {
                let stop = rms(&to_16k(&tone(sr, 9000.0, 1.0), sr).unwrap());
                assert!(stop < 1e-3, "{sr}: 9 kHz leaks {stop}");
            }
        }
        assert!(to_16k(&[0.0; 10], 100).is_err());
        // 767,999 Hz has 16,000 phases of 9,460 taps: filters computed per sample instead of tabulated
        assert_eq!(to_16k(&[0.0; 96], 767_999).unwrap().len(), 3);
    }

    #[test]
    fn features_of_silence_and_frames() {
        let mel = Mel::new(80);
        let (m, frames) = mel.features(&vec![0.0; 16000]);
        assert_eq!(m.len(), 80 * N_FRAMES);
        assert_eq!(frames, 100);
        assert!(m.iter().all(|&v| (v - (-10.0 + 4.0) / 4.0).abs() < 1e-6)); // log10 floor, then (x + 4) / 4
        let (m, _) = mel.features(&tone(16000, 1000.0, 1.0));
        // the 1 kHz tone lands in one mel band, the highest value of the first frame row
        let col: Vec<f32> = (0..80).map(|b| m[b * N_FRAMES + 50]).collect();
        let peak = col.iter().cloned().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        assert!((20..40).contains(&peak), "peak band {peak}");
    }
}
