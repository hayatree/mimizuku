//! The transcription pipeline: features of every clip, then encoding and greedy decoding batch by batch, with
//! token timestamps from the alignment heads.
use crate::gpu::R;
use crate::model::{self, TENC};
use crate::{align, audio, dev};
use cudarc::driver::sys;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

pub struct Clip {
    pub path: String,
    pub mel: Vec<f32>,
    /// 10 ms frames that hold audio
    pub frames: usize,
}

/// One clip's result: language token, the whole token sequence (prompt included) and each token's start time.
pub struct Decoded {
    pub lang: i32,
    pub seq: Vec<i32>,
    pub ts: Vec<f32>,
}

/// WAV -> 16 kHz -> log-mel for every file, on all cores.
pub fn features(paths: &[String], mel: &audio::Mel) -> R<Vec<Clip>> {
    let next = AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Result<Clip, String>>>> = (0..paths.len()).map(|_| std::sync::Mutex::new(None)).collect();
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(paths.len().max(1));
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let r = (|| -> Result<Clip, String> {
                        let (y, sr) = audio::read_wav(&paths[i]).map_err(|e| e.to_string())?;
                        let y = audio::to_16k(&y, sr).map_err(|e| format!("{}: {e}", paths[i]))?;
                        if y.len() > audio::N_SAMPLES {
                            return Err(format!(
                                "{}: {:.1} s is longer than Whisper's 30 s window; split it into shorter clips",
                                paths[i],
                                y.len() as f64 / audio::SR as f64
                            ));
                        }
                        let (m, frames) = mel.features(&y);
                        Ok(Clip { path: paths[i].clone(), mel: m, frames })
                    })();
                    *slots[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    Ok(slots.into_iter().map(|s| s.into_inner().unwrap().unwrap()).collect::<Result<_, String>>()?)
}

/// The decoder prompt: start, language, task, no timestamps (English-only models: start, no timestamps).
pub struct Prompt {
    multilingual: bool,
    sot: i32,
    task: i32,
    no_timestamps: i32,
    /// the --language token; None detects the language per clip
    lang: Option<i32>,
}

impl Prompt {
    pub fn new(gc: &model::GenCfg, language: &str, task: &str) -> R<Self> {
        let multilingual = !gc.langs.is_empty();
        if !multilingual {
            if !matches!(language, "auto" | "en") || task != "transcribe" {
                return Err("this English-only model takes only --language en (or auto) and --task transcribe".into());
            }
            return Ok(Prompt { multilingual, sot: gc.sot, task: -1, no_timestamps: gc.no_timestamps, lang: None });
        }
        let lang = match language {
            "auto" => None,
            code => Some(gc.lang_token(code).ok_or(format!("--language {code}: not a language of this model"))?),
        };
        let task = gc.tasks.iter().find(|(n, _)| n == task).map(|&(_, t)| t).ok_or(format!("--task {task}: unknown task"))?;
        Ok(Prompt { multilingual, sot: gc.sot, task, no_timestamps: gc.no_timestamps, lang })
    }

    /// `lang` stands at position 1 until detection overwrites it.
    fn tokens(&self, lang: i32) -> Vec<i32> {
        if self.multilingual { vec![self.sot, lang, self.task, self.no_timestamps] } else { vec![self.sot, self.no_timestamps] }
    }

    fn detect(&self) -> bool {
        self.multilingual && self.lang.is_none()
    }
}

#[derive(Default)]
pub struct Times {
    pub enc: f64,
    pub dec: f64,
    pub post: f64,
    pub steps: usize,
}

/// Encodes and decodes up to `e.bmax` clips at once; returns their tokens with timestamps.
pub fn transcribe(e: &mut model::Engine, chunk: &[Clip], p: &Prompt, graph: bool, reference: Option<&dev::Reference>, tm: &mut Times) -> R<Vec<Decoded>> {
    let (b, eg) = (chunk.len(), e.eg);
    let t0 = Instant::now();
    for (gi, grp) in chunk.chunks(eg).enumerate() {
        let mels: Vec<&[f32]> = grp.iter().map(|c| &c.mel[..]).collect();
        e.encode(&mels, gi * eg)?;
        if let Some(r) = reference {
            r.encoder(e)?;
        }
    }
    e.g.sync()?;
    tm.enc += t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let np = p.tokens(0).len();
    let (mt, eos) = (e.dims.max_t, e.gc.eos);
    e.reset(&vec![p.tokens(p.lang.unwrap_or(p.sot)); b])?;
    let langs = if p.detect() { e.detect(b)? } else { vec![p.lang.unwrap_or(-1); b] };
    let mut steps = p.detect() as usize;
    let graph = if graph {
        e.g.stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        e.step(b, false, np)?;
        let gr = e.g.stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
        let gr = gr.ok_or("empty graph")?;
        gr.upload()?;
        Some(gr)
    } else {
        None
    };
    while steps < mt - 1 {
        match &graph {
            Some(gr) => gr.launch()?,
            None => e.step(b, false, np)?,
        }
        steps += 1;
        if steps.is_multiple_of(8) && steps >= np && e.g.download::<i32>(e.done, b)?.iter().all(|&d| d != 0) {
            break;
        }
    }
    e.g.sync()?;
    tm.dec += t0.elapsed().as_secs_f64();
    tm.steps += steps;

    let t0 = Instant::now();
    let toks = e.g.download::<i32>(e.tok, b * mt)?;
    let row = e.gc.align_heads.len() * TENC;
    let mut out = vec![];
    for (s, c) in chunk.iter().enumerate() {
        let t = &toks[s * mt..(s + 1) * mt];
        let n_seq = t.iter().skip(np).position(|&x| x == eos).map(|k| k + np + 1).unwrap_or(steps + 1).min(mt);
        // the alignment rows of the generated tokens fed back as input; the last token takes the time of the one before
        let rows = (n_seq - 1).saturating_sub(np);
        let mut ts = vec![0f32; n_seq];
        if rows > 0 {
            let w = e.g.download_h(e.align + 2 * ((s * mt + np) * row) as u64, rows * row)?;
            let jt = align::jump_times(&w, rows, e.gc.align_heads.len(), TENC, c.frames / 2, 7);
            ts[np..np + rows].copy_from_slice(&jt);
            ts[n_seq - 1] = jt[rows - 1];
        }
        out.push(Decoded { lang: langs[s], seq: t[..n_seq].to_vec(), ts });
    }
    tm.post += t0.elapsed().as_secs_f64();
    Ok(out)
}
