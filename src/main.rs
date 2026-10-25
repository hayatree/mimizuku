//! mimizuku: batched Whisper transcription with token timestamps on a CUDA GPU.
//! See README.md for the options; `--help` prints them.
mod align;
mod audio;
mod dev;
mod gpu;
mod json;
mod model;
mod st;
mod tok;
mod transcribe;

use gpu::R;
use std::time::Instant;
use transcribe::{Clip, Decoded, Prompt, Times};

const USAGE: &str = "\
usage: mimizuku --model DIR [options] FILE.wav...
       mimizuku --model DIR [options] --dir DIR

Transcribes WAV clips of up to 30 s with a Whisper checkpoint in Hugging Face format (model.safetensors,
config.json, generation_config.json, vocab.json), greedy decoding, with a start and end time for every
character or word piece. Writes JSON: {path: {\"language\", \"text\", \"words\": [[piece, start, end], ...]}}.

options:
  --model DIR        model directory (required)
  --dir DIR          every .wav file in DIR, sorted by name
  --prefix STR       with --dir: only files whose name starts with STR
  --language CODE    language code such as en or ja; auto (default) detects it per clip
  --task NAME        transcribe (default) or translate (into English)
  --out FILE         write the JSON to FILE instead of standard output
  --batch N          clips decoded together (default 32); lower it if GPU memory is short
  --enc-group N      clips per encoder pass (default 8)
  --quiet            no progress or timing on standard error
development:
  --ref FILE         check every stage against a reference from scripts/hf_reference.py
  --enc-attn cublas  unfused encoder attention (cuBLAS scores, softmax, cuBLAS P.V) for comparison
  --no-graph         launch every decoding step's kernels instead of replaying a CUDA graph
";

struct Args {
    opts: Vec<(String, String)>,
    flags: Vec<String>,
    files: Vec<String>,
}

impl Args {
    fn parse() -> R<Self> {
        const WITH_VALUE: &[&str] = &["--model", "--dir", "--prefix", "--language", "--task", "--out", "--batch", "--enc-group", "--ref", "--enc-attn"];
        const FLAGS: &[&str] = &["--quiet", "--no-graph", "--help", "-h"];
        let mut a = Args { opts: vec![], flags: vec![], files: vec![] };
        let mut it = std::env::args().skip(1);
        while let Some(s) = it.next() {
            if WITH_VALUE.contains(&s.as_str()) {
                let v = it.next().ok_or(format!("{s} needs a value"))?;
                a.opts.push((s, v));
            } else if FLAGS.contains(&s.as_str()) {
                a.flags.push(s);
            } else if s.starts_with('-') {
                return Err(format!("unknown option {s} (see --help)").into());
            } else {
                a.files.push(s);
            }
        }
        Ok(a)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.iter().rev().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
    fn usize(&self, k: &str, default: usize) -> R<usize> {
        match self.get(k) {
            Some(v) => v.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| format!("{k}: expected a positive integer, got {v}").into()),
            None => Ok(default),
        }
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o += "\\\"",
            '\\' => o += "\\\\",
            '\n' => o += "\\n",
            '\r' => o += "\\r",
            '\t' => o += "\\t",
            c if (c as u32) < 0x20 => o += &format!("\\u{:04x}", c as u32),
            c => o.push(c),
        }
    }
    o + "\""
}

/// Python's round(x, 2) printed the way json.dumps prints a float.
fn json_t(x: f32) -> String {
    let v: f64 = format!("{:.2}", x as f64).parse().unwrap();
    format!("{v:?}")
}

fn list_dir(d: &str, prefix: &str) -> R<Vec<String>> {
    let mut v: Vec<String> = std::fs::read_dir(d)
        .map_err(|e| format!("{d}: {e}"))?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .filter(|n| n.starts_with(prefix) && n.to_ascii_lowercase().ends_with(".wav"))
        .map(|n| format!("{}/{n}", d.trim_end_matches('/')))
        .collect();
    v.sort();
    Ok(v)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn to_json(clips: &[Clip], results: &[Decoded], tk: &tok::Tok, gc: &model::GenCfg) -> String {
    let mut out = String::from("{");
    for (i, (c, r)) in clips.iter().zip(results).enumerate() {
        let ws = tk.pieces(&r.seq, &r.ts, gc.eos);
        let text = tk.decode(&r.seq.iter().copied().filter(|&t| t < gc.eos).collect::<Vec<_>>());
        let wj: Vec<String> = ws.iter().map(|(w, a, b)| format!("[{}, {}, {}]", json_str(w), json_t(*a), json_t(*b))).collect();
        let lang = if gc.langs.is_empty() { "en" } else { gc.lang_code(r.lang).unwrap_or("") };
        out += &format!(
            "{}{}: {{\"language\": {}, \"text\": {}, \"words\": [{}]}}",
            if i > 0 { ", " } else { "" },
            json_str(&c.path),
            json_str(lang),
            json_str(&text),
            wj.join(", ")
        );
    }
    out + "}\n"
}

fn run() -> R<()> {
    let args = Args::parse()?;
    if args.flag("--help") || args.flag("-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let quiet = args.flag("--quiet");
    let log = |s: String| {
        if !quiet {
            eprintln!("{s}");
        }
    };
    let model_dir = args.get("--model").ok_or("--model is required (see --help)")?.to_string();
    let reference = args.get("--ref").map(dev::Reference::open).transpose()?;
    let t_start = Instant::now();

    let mut paths: Vec<String> = args.files.clone();
    if let Some(r) = &reference {
        paths = vec![r.wav()?];
    } else if let Some(d) = args.get("--dir") {
        paths.extend(list_dir(d, args.get("--prefix").unwrap_or(""))?);
    }
    if paths.is_empty() {
        return Err("no input files (see --help)".into());
    }
    let dims = model::Dims::load(&model_dir)?;
    let bmax = args.usize("--batch", 32)?.min(paths.len());
    let eg = args.usize("--enc-group", 8)?.min(bmax);

    let t0 = Instant::now();
    let mel = audio::Mel::new(dims.n_mels);
    let mut clips = transcribe::features(&paths, &mel)?;
    let t_feat = t0.elapsed().as_secs_f64();
    if let Some(r) = &reference {
        r.features(&mel, &mut clips[0])?;
    }

    let t0 = Instant::now();
    let mut e = model::Engine::new(&model_dir, bmax, eg)?;
    e.fused_attn = args.get("--enc-attn") != Some("cublas");
    if reference.as_ref().is_some_and(|r| r.has_trace()) {
        e.trace = Some(vec![]);
    }
    let tk = tok::Tok::load(&model_dir, e.dims.vocab, e.gc.eos)?;
    let t_load = t0.elapsed().as_secs_f64();
    log(format!(
        "model on the GPU: {:.1} GB allocated in {t_load:.1}s (CUDA setup {:.1}s, weights {:.1}s)",
        e.g.allocated as f64 / 1e9,
        e.load_times.0,
        e.load_times.1
    ));

    let prompt = Prompt::new(&e.gc, args.get("--language").unwrap_or("auto"), args.get("--task").unwrap_or("transcribe"))?;
    let mut tm = Times::default();
    let mut results = vec![];
    for chunk in clips.chunks(bmax) {
        let (dec0, steps0) = (tm.dec, tm.steps);
        results.extend(transcribe::transcribe(&mut e, chunk, &prompt, !args.flag("--no-graph"), reference.as_ref(), &mut tm)?);
        let (dt, steps) = (tm.dec - dec0, tm.steps - steps0);
        log(format!("decoded {} clips: {steps} steps in {dt:.2}s = {:.1} ms/step", chunk.len(), 1000.0 * dt / steps as f64));
    }
    if let Some(r) = &reference {
        r.sequence(&results[0])?;
    }

    let t0 = Instant::now();
    let out = to_json(&clips, &results, &tk, &e.gc);
    match args.get("--out") {
        Some(p) => std::fs::write(p, &out).map_err(|e| format!("{p}: {e}"))?,
        None => print!("{out}"),
    }
    tm.post += t0.elapsed().as_secs_f64();
    let audio_s: f64 = clips.iter().map(|c| c.frames as f64 / 100.0).sum();
    log(format!(
        "{} clips ({audio_s:.1}s of audio), batch {bmax}, encoder group {eg}: features {t_feat:.2}s | model load {t_load:.1}s | encoder {:.2}s | decode {:.2}s ({} steps) | timestamps+text {:.2}s | without load {:.2}s | total {:.1}s",
        clips.len(),
        tm.enc,
        tm.dec,
        tm.steps,
        tm.post,
        t_feat + tm.enc + tm.dec + tm.post,
        t_start.elapsed().as_secs_f64()
    ));
    Ok(())
}
