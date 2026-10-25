//! `--ref`: checks every stage against a reference saved by scripts/hf_reference.py.
use crate::gpu::R;
use crate::model::{Engine, TENC};
use crate::transcribe::{Clip, Decoded};
use crate::{audio, st};

/// Relative L2 error and max abs error of a against reference r (of the same length).
fn cmp(a: &[f32], r: &[f32]) -> R<(f64, f64)> {
    if a.len() != r.len() {
        return Err(format!("comparing {} values with a reference of {}", a.len(), r.len()).into());
    }
    let (mut d2, mut r2, mut mx) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(r) {
        let d = (*x - *y) as f64;
        d2 += d * d;
        r2 += (*y as f64) * (*y as f64);
        mx = mx.max(d.abs());
    }
    Ok(((d2 / r2.max(1e-30)).sqrt(), mx))
}

fn report(name: &str, a: &[f32], r: &[f32]) -> R<()> {
    let (rel, mx) = cmp(a, r)?;
    eprintln!("verify {name:44} rel {rel:.2e} max {mx:.3e}");
    Ok(())
}

pub struct Reference(st::File);

impl Reference {
    pub fn open(path: &str) -> R<Self> {
        Ok(Reference(st::File::open(path)?))
    }

    pub fn wav(&self) -> R<String> {
        Ok(self.0.meta.get("wav").cloned().ok_or("ref: no wav in metadata")?)
    }

    /// Whether the reference holds every encoder layer's input (then the engine should record them).
    pub fn has_trace(&self) -> bool {
        self.0.has("enc_hidden")
    }

    /// Log-mel features from the reference's 16 kHz audio and from the WAV through this engine's resampler; the
    /// network then gets exactly the reference features.
    pub fn features(&self, mel: &audio::Mel, clip: &mut Clip) -> R<()> {
        let (rf, _) = self.0.f32s("features")?;
        let (a16, _) = self.0.f32s("audio16k")?;
        report("features from the reference 16 kHz audio", &mel.features(&a16).0, &rf)?;
        report("features from the WAV (own resampler)", &clip.mel, &rf)?;
        let nf = *self.0.i32s("num_frames")?.0.first().ok_or("ref: num_frames is empty")?;
        eprintln!("verify audio frames: engine {} reference {nf}", clip.frames);
        clip.mel = rf;
        Ok(())
    }

    pub fn encoder(&self, e: &mut Engine) -> R<()> {
        let n = TENC * e.dims.d;
        report("encoder output", &e.g.download_h(e.eh, n)?, &self.0.floats("encoder")?.0)?;
        if let Some(t) = e.trace.take() {
            let (rh, _) = self.0.floats("enc_hidden")?;
            if rh.len() != t.len() * n {
                return Err(format!("ref: enc_hidden has {} values, expected {} layers of {n}", rh.len(), t.len()).into());
            }
            for (i, (a, r)) in t.iter().zip(rh.chunks(n)).enumerate() {
                report(&format!("encoder layer {i:2} input"), a, r)?;
            }
        }
        Ok(())
    }

    /// Tokens must be identical; timestamp differences are reported.
    pub fn sequence(&self, got: &Decoded) -> R<()> {
        let (rs, _) = self.0.i32s("sequence")?;
        let (rt, _) = self.0.f32s("token_timestamps")?;
        if got.ts.len() != rt.len() || rt.iter().any(|t| !t.is_finite()) {
            return Err(format!("ref: {} token timestamps, the engine has {}", rt.len(), got.ts.len()).into());
        }
        let n = rt.len();
        let diffs: Vec<f32> = got.ts.iter().zip(&rt).map(|(a, b)| (a - b).abs()).collect();
        eprintln!(
            "verify token timestamps: {} of {n} identical, max diff {:.2}s",
            diffs.iter().filter(|&&d| d < 1e-4).count(),
            diffs.iter().copied().fold(0.0, f32::max)
        );
        eprintln!("verify sequence: {} tokens, reference {}, identical: {}", got.seq.len(), rs.len(), got.seq == rs);
        if got.seq != rs {
            eprintln!("  engine    {:?}\n  reference {rs:?}", got.seq);
            return Err("the token sequence differs from the reference".into());
        }
        Ok(())
    }
}
