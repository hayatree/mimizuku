//! Whisper (any size, fp16) on the GPU: encoder for groups of clips, cross-attention k/v for all decoder layers,
//! and one greedy decoding step for B sequences at once (captured as a CUDA graph by the caller).
use crate::gpu::{A, Arena, Gpu, Mm, Ptr, R, f2h};
use crate::json::{self, Json};
use crate::st;

pub const HD: usize = 64;
const NFRAMES: usize = crate::audio::N_FRAMES;
/// Encoder positions: the conv stem halves the mel frames (TENC in kernels/whisper.cu too).
pub const TENC: usize = NFRAMES / 2;
const NCH: usize = TENC.div_ceil(64);
const EPS: f32 = 1e-5;

/// Model dimensions from config.json.
#[derive(Clone, Debug)]
pub struct Dims {
    pub d: usize,
    pub ff: usize,
    pub nh: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub n_mels: usize,
    pub vocab: usize,
    pub max_t: usize,
}

fn read_json(path: &str) -> R<Json> {
    json::parse_json(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
}

impl Dims {
    pub fn load(model_dir: &str) -> R<Self> {
        let c = read_json(&format!("{model_dir}/config.json"))?;
        let n = |k: &str| -> R<usize> {
            c.get(k).and_then(|v| v.uint()).filter(|&n| (1..1 << 20).contains(&n)).ok_or_else(|| format!("config.json: {k} missing or out of range").into())
        };
        let d = Dims {
            d: n("d_model")?,
            ff: n("encoder_ffn_dim")?,
            nh: n("encoder_attention_heads")?,
            enc_layers: n("encoder_layers")?,
            dec_layers: n("decoder_layers")?,
            n_mels: n("num_mel_bins")?,
            vocab: n("vocab_size")?,
            max_t: n("max_target_positions")?,
        };
        if d.nh * HD != d.d || n("decoder_attention_heads")? != d.nh || n("decoder_ffn_dim")? != d.ff {
            return Err(format!("unsupported shape {d:?}: the kernels need 64-wide heads and equal encoder/decoder widths").into());
        }
        // gemv kernels load 8 halves at a time; the prompt takes up to 4 positions
        if n("max_source_positions")? != TENC
            || !(16..=448).contains(&d.max_t)
            || !d.d.is_multiple_of(128)
            || !d.ff.is_multiple_of(8)
            || d.enc_layers > 64
            || d.dec_layers > 64
        {
            return Err(format!("unsupported shape {d:?}").into());
        }
        Ok(d)
    }
}

/// Prompt and decoding settings from generation_config.json.
pub struct GenCfg {
    pub sot: i32,
    pub eos: i32,
    pub no_timestamps: i32,
    /// language code ("ja") and token, empty for English-only models
    pub langs: Vec<(String, i32)>,
    pub tasks: Vec<(String, i32)>,
    pub suppress: Vec<usize>,
    pub begin_suppress: Vec<i32>,
    /// (decoder layer, head) pairs whose cross-attention gives the token timestamps
    pub align_heads: Vec<(usize, usize)>,
}

impl GenCfg {
    pub fn load(model_dir: &str, dims: &Dims) -> R<Self> {
        let g = read_json(&format!("{model_dir}/generation_config.json"))?;
        let token = |v: &Json| v.uint().filter(|&t| t < dims.vocab).map(|t| t as i32);
        let id = |k: &str| -> R<i32> { g.get(k).and_then(token).ok_or_else(|| format!("generation_config.json: {k} missing or not a token id").into()) };
        let map = |k: &str, strip: bool| -> R<Vec<(String, i32)>> {
            match g.get(k) {
                None => Ok(vec![]),
                Some(Json::Obj(items)) => items
                    .iter()
                    .map(|(s, v)| {
                        let s = if strip { s.trim_start_matches("<|").trim_end_matches("|>").to_string() } else { s.clone() };
                        match token(v) {
                            Some(t) => Ok((s, t)),
                            None => Err(format!("generation_config.json: {k}.{s} is not a token id").into()),
                        }
                    })
                    .collect(),
                Some(_) => Err(format!("generation_config.json: {k} is not an object").into()),
            }
        };
        let align_heads: Vec<(usize, usize)> = match g.get("alignment_heads") {
            // openai/whisper's default: every head of the second half of the decoder layers
            None => (dims.dec_layers / 2..dims.dec_layers).flat_map(|l| (0..dims.nh).map(move |h| (l, h))).collect(),
            Some(Json::Arr(v)) => v
                .iter()
                .map(|lh| match lh.uints().as_deref() {
                    Some(&[l, h]) if l < dims.dec_layers && h < dims.nh => Some((l, h)),
                    _ => None,
                })
                .collect::<Option<_>>()
                .ok_or("generation_config.json: alignment_heads must be [layer, head] pairs within the model")?,
            Some(_) => return Err("generation_config.json: alignment_heads is not a list".into()),
        };
        let mut unique = align_heads.clone();
        unique.sort();
        unique.dedup();
        if align_heads.is_empty() || unique.len() != align_heads.len() {
            return Err("generation_config.json: alignment_heads is empty or has duplicates".into());
        }
        let multilingual = g.get("is_multilingual").and_then(|j| j.bool()).unwrap_or(true);
        let ids = |k: &str| -> R<Vec<i32>> {
            match g.get(k) {
                None => Ok(vec![]),
                Some(Json::Arr(v)) => {
                    v.iter().map(token).collect::<Option<_>>().ok_or_else(|| format!("generation_config.json: {k} has an invalid token id").into())
                }
                Some(_) => Err(format!("generation_config.json: {k} is not a list").into()),
            }
        };
        let begin_suppress = ids("begin_suppress_tokens")?;
        if begin_suppress.len() > 2 {
            return Err("generation_config.json: more than two begin_suppress_tokens are not supported".into());
        }
        let langs = if multilingual { map("lang_to_id", true)? } else { vec![] };
        let tasks = map("task_to_id", false)?;
        if multilingual && (langs.is_empty() || tasks.is_empty()) {
            return Err("generation_config.json: a multilingual model needs lang_to_id and task_to_id".into());
        }
        Ok(GenCfg {
            sot: id("decoder_start_token_id")?,
            eos: id("eos_token_id")?,
            no_timestamps: id("no_timestamps_token_id")?,
            langs,
            tasks,
            suppress: ids("suppress_tokens")?.into_iter().map(|t| t as usize).collect(),
            begin_suppress,
            align_heads,
        })
    }

    pub fn lang_token(&self, code: &str) -> Option<i32> {
        self.langs.iter().find(|(c, _)| c == code).map(|&(_, t)| t)
    }

    pub fn lang_code(&self, tok: i32) -> Option<&str> {
        self.langs.iter().find(|&&(_, t)| t == tok).map(|(c, _)| c.as_str())
    }
}

struct Ln {
    w: Ptr,
    b: Ptr,
}

/// Linear layer: weight [n, k] and bias [n] (zeros where the checkpoint has none, e.g. k_proj).
struct Lin {
    w: Ptr,
    b: Ptr,
}

struct EncLayer {
    ln1: Ln,
    qkv: Lin,
    o: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
}

/// One decoder layer's linear outputs, [bmax][n] each.
struct DecOut {
    qkv: Ptr,
    o: Ptr,
    xq: Ptr,
    xo: Ptr,
    fc1: Ptr,
    fc2: Ptr,
}

struct DecLayer {
    ln1: Ln,
    qkv: Lin,
    o: Lin,
    lnx: Ln,
    xq: Lin,
    xo: Lin,
    ln3: Ln,
    fc1: Lin,
    fc2: Lin,
}

/// Places checkpoint tensors (F16, or F32 converted to F16) in the weight arena.
struct Loader<'a> {
    f: &'a st::File,
    arena: Arena,
}

enum Part<'a> {
    T(&'a str),
    Zeros(usize),
}

/// A tensor's bytes as fp16.
fn f16_bytes<'f>(f: &'f st::File, name: &str) -> R<std::borrow::Cow<'f, [u8]>> {
    let t = f.get(name)?;
    match t.dtype.as_str() {
        "F16" => Ok(std::borrow::Cow::Borrowed(t.data)),
        "F32" => Ok(std::borrow::Cow::Owned(t.data.as_chunks::<4>().0.iter().flat_map(|&c| f2h(f32::from_le_bytes(c)).to_le_bytes()).collect())),
        dt => Err(format!("{name}: unsupported dtype {dt}").into()),
    }
}

impl Loader<'_> {
    fn check(&self, name: &str, want: &[usize]) -> R<()> {
        let t = self.f.get(name)?;
        if t.shape != want {
            return Err(format!("model.safetensors: {name} has shape {:?}, expected {want:?}", t.shape).into());
        }
        Ok(())
    }
    /// Tensors concatenated along the first axis into one buffer.
    fn cat(&mut self, g: &Gpu, parts: &[Part]) -> R<Ptr> {
        let f = self.f;
        let bufs = parts
            .iter()
            .map(|p| match p {
                Part::T(name) => f16_bytes(f, name),
                Part::Zeros(n) => Ok(std::borrow::Cow::Owned(vec![0u8; 2 * n])),
            })
            .collect::<R<Vec<_>>>()?;
        let refs: Vec<&[u8]> = bufs.iter().map(|b| &b[..]).collect();
        self.arena.put(g, &refs)
    }
    fn t(&mut self, g: &Gpu, name: &str, shape: &[usize]) -> R<Ptr> {
        self.check(name, shape)?;
        self.cat(g, &[Part::T(name)])
    }
    fn ln(&mut self, g: &Gpu, p: &str, d: usize) -> R<Ln> {
        Ok(Ln { w: self.t(g, &format!("{p}.weight"), &[d])?, b: self.t(g, &format!("{p}.bias"), &[d])? })
    }
    /// A linear or convolution layer whose weight has shape `w` (output channels first).
    fn lin(&mut self, g: &Gpu, p: &str, w: &[usize]) -> R<Lin> {
        Ok(Lin { w: self.t(g, &format!("{p}.weight"), w)?, b: self.t(g, &format!("{p}.bias"), &w[..1])? })
    }
    /// q, k, v projections fused into one [3D, D] weight; k has no bias.
    fn qkv(&mut self, g: &Gpu, p: &str, d: usize) -> R<Lin> {
        let n = |s: &str| format!("{p}.{s}");
        let (qw, kw, vw, qb, vb) = (n("q_proj.weight"), n("k_proj.weight"), n("v_proj.weight"), n("q_proj.bias"), n("v_proj.bias"));
        for w in [&qw, &kw, &vw] {
            self.check(w, &[d, d])?;
        }
        for b in [&qb, &vb] {
            self.check(b, &[d])?;
        }
        Ok(Lin { w: self.cat(g, &[Part::T(&qw), Part::T(&kw), Part::T(&vw)])?, b: self.cat(g, &[Part::T(&qb), Part::Zeros(d), Part::T(&vb)])? })
    }
}

pub struct Engine {
    pub g: Gpu,
    pub dims: Dims,
    pub gc: GenCfg,
    enc: Vec<EncLayer>,
    dec: Vec<DecLayer>,
    conv1: Lin,
    conv2: Lin,
    epos: Ptr,
    eln: Ln,
    xkv: Lin, // every decoder layer's cross k/v projection fused: [dec_layers * 2D, D]
    emb: Ptr,
    dpos: Ptr,
    dln: Ln,
    pub bmax: usize,
    pub eg: usize,
    /// seconds spent setting up CUDA, and reading the checkpoint and placing it on the GPU
    pub load_times: (f64, f64),
    /// flash attention in the encoder (otherwise cuBLAS scores + softmax + cuBLAS P.V)
    pub fused_attn: bool,
    /// keep a copy of every encoder layer's input (verification only)
    pub trace: Option<Vec<Vec<f32>>>,
    // encoder scratch for eg clips
    mel: Ptr,
    x1: Ptr,
    h1: Ptr,
    x2: Ptr,
    ex: Ptr,
    pub eh: Ptr,
    eqkv: Ptr,
    esc: Ptr,
    eatt: Ptr,
    ef: Ptr,
    // decoder state for bmax sequences
    xkv_buf: Ptr,
    pub tok: Ptr,
    pub done: Ptr,
    ctr: Ptr,
    x: Ptr,
    hb: Ptr,
    att: Ptr,
    /// every decoder layer's linear outputs for bmax rows (DecOut), and the same layout holding the biases:
    /// copied over the outputs at the start of a step, so the GEMMs accumulate onto them without one fill per GEMM
    dout: Ptr,
    dbias: Ptr,
    logits: Ptr,
    kc: Ptr,
    vc: Ptr,
    s: Ptr,
    stat: Ptr,
    part: Ptr,
    pub align: Ptr,
    aslot: Ptr,
    sup: Ptr,
    sup_lang: Ptr,
}

/// Device bytes the engine allocates besides the weights, for bmax sequences and eg clips per encoder group.
pub fn scratch_bytes(dm: &Dims, na: usize, bmax: usize, eg: usize) -> usize {
    let (d, ff) = (dm.d, dm.ff);
    let enc = eg * (dm.n_mels * NFRAMES + NFRAMES * 3 * dm.n_mels + NFRAMES * d + TENC * (3 * d + d + d + 3 * d + d + ff)) + dm.nh * TENC * TENC;
    let dec = bmax * (2 * dm.dec_layers * (d * TENC + dm.max_t * d + 7 * d + ff) + dm.max_t * na * TENC + 4 * d + dm.vocab);
    2 * (enc + dec) + bmax * dm.nh * (TENC + NCH * (2 + HD)) * 4 + (128 << 20)
}

impl Engine {
    pub fn new(model_dir: &str, bmax: usize, eg: usize) -> R<Self> {
        let dims = Dims::load(model_dir)?;
        let gc = GenCfg::load(model_dir, &dims)?;
        let (d, ff, nh, nl) = (dims.d, dims.ff, dims.nh, dims.dec_layers);
        let na = gc.align_heads.len();
        let t0 = std::time::Instant::now();
        let mut g = Gpu::new()?;
        let t_init = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let f = st::File::open(&format!("{model_dir}/model.safetensors"))?;
        let weights = f.f16_bytes() + 256 * (f.tensor_count() + 2) + 2 * (dims.enc_layers + 2 * nl) * d;
        let need = weights + scratch_bytes(&dims, na, bmax, eg);
        // a warning only: on unified-memory systems (DGX Spark, Jetson) the page cache counts as used but is reclaimable
        let (free, total) = g.mem_info()?;
        let gb = |b: usize| b as f64 / 1e9;
        if need > total {
            return Err(format!(
                "not enough GPU memory: about {:.1} GB needed for --batch {bmax} --enc-group {eg}, {:.1} GB in total; use a smaller --batch",
                gb(need),
                gb(total)
            )
            .into());
        } else if need > free {
            eprintln!("warning: about {:.1} GB of GPU memory needed, {:.1} GB reported free; if allocation fails, use a smaller --batch", gb(need), gb(free));
        }
        let arena = Arena::new(&mut g, weights)?;
        let mut l = Loader { f: &f, arena };
        let mut enc = vec![];
        for i in 0..dims.enc_layers {
            let p = format!("model.encoder.layers.{i}");
            enc.push(EncLayer {
                ln1: l.ln(&g, &format!("{p}.self_attn_layer_norm"), d)?,
                qkv: l.qkv(&g, &format!("{p}.self_attn"), d)?,
                o: l.lin(&g, &format!("{p}.self_attn.out_proj"), &[d, d])?,
                ln2: l.ln(&g, &format!("{p}.final_layer_norm"), d)?,
                fc1: l.lin(&g, &format!("{p}.fc1"), &[ff, d])?,
                fc2: l.lin(&g, &format!("{p}.fc2"), &[d, ff])?,
            });
        }
        let mut dec = vec![];
        let (mut xw, mut xb) = (vec![], vec![]);
        for i in 0..nl {
            let p = format!("model.decoder.layers.{i}");
            dec.push(DecLayer {
                ln1: l.ln(&g, &format!("{p}.self_attn_layer_norm"), d)?,
                qkv: l.qkv(&g, &format!("{p}.self_attn"), d)?,
                o: l.lin(&g, &format!("{p}.self_attn.out_proj"), &[d, d])?,
                lnx: l.ln(&g, &format!("{p}.encoder_attn_layer_norm"), d)?,
                xq: l.lin(&g, &format!("{p}.encoder_attn.q_proj"), &[d, d])?,
                xo: l.lin(&g, &format!("{p}.encoder_attn.out_proj"), &[d, d])?,
                ln3: l.ln(&g, &format!("{p}.final_layer_norm"), d)?,
                fc1: l.lin(&g, &format!("{p}.fc1"), &[ff, d])?,
                fc2: l.lin(&g, &format!("{p}.fc2"), &[d, ff])?,
            });
            xw.push(format!("{p}.encoder_attn.k_proj.weight"));
            xw.push(format!("{p}.encoder_attn.v_proj.weight"));
            xb.push(format!("{p}.encoder_attn.v_proj.bias"));
        }
        for w in &xw {
            l.check(w, &[d, d])?;
        }
        for b in &xb {
            l.check(b, &[d])?;
        }
        let xw_parts: Vec<Part> = xw.iter().map(|s| Part::T(s)).collect();
        let xb_parts: Vec<Part> = xb.iter().flat_map(|s| [Part::Zeros(d), Part::T(s)]).collect();
        let xkv = Lin { w: l.cat(&g, &xw_parts)?, b: l.cat(&g, &xb_parts)? };
        let conv1 = l.lin(&g, "model.encoder.conv1", &[d, dims.n_mels, 3])?; // = [D, 3 n_mels], column ci * 3 + k
        let conv2 = l.lin(&g, "model.encoder.conv2", &[d, d, 3])?;
        let epos = l.t(&g, "model.encoder.embed_positions.weight", &[TENC, d])?;
        let eln = l.ln(&g, "model.encoder.layer_norm", d)?;
        let emb = l.t(&g, "model.decoder.embed_tokens.weight", &[dims.vocab, d])?;
        let dpos = l.t(&g, "model.decoder.embed_positions.weight", &[dims.max_t, d])?;
        let dln = l.ln(&g, "model.decoder.layer_norm", d)?;
        l.arena.finish(&g)?;
        drop(f);
        let load_times = (t_init, t0.elapsed().as_secs_f64());

        let mut supmask = vec![0u8; dims.vocab];
        for &t in &gc.suppress {
            supmask[t] = 1;
        }
        let sup = g.upload_new(&supmask)?;
        let mut langmask = vec![1u8; dims.vocab];
        for &(_, t) in &gc.langs {
            *langmask.get_mut(t as usize).ok_or("generation_config.json: language token out of range")? = 0;
        }
        let sup_lang = g.upload_new(&langmask)?;
        let mut slots = vec![-1i32; nl * nh];
        for (s, &(layer, head)) in gc.align_heads.iter().enumerate() {
            slots[layer * nh + head] = s as i32;
        }
        let aslot = g.upload_new(&slots)?;

        let h = |g: &mut Gpu, n: usize| g.alloc_h(n);
        let (r1, r2, mt) = (eg * NFRAMES, eg * TENC, dims.max_t);
        let mel = h(&mut g, eg * dims.n_mels * NFRAMES)?;
        let x1 = h(&mut g, r1 * 3 * dims.n_mels)?;
        let h1 = h(&mut g, r1 * d)?;
        let x2 = h(&mut g, r2 * 3 * d)?;
        let ex = h(&mut g, r2 * d)?;
        let eh = h(&mut g, r2 * d)?;
        let eqkv = h(&mut g, r2 * 3 * d)?;
        let esc = h(&mut g, nh * TENC * TENC)?;
        let eatt = h(&mut g, r2 * d)?;
        let ef = h(&mut g, r2 * ff)?;
        let b = bmax;
        let xkv_buf = h(&mut g, b * nl * 2 * d * TENC)?;
        let tok = g.alloc(b * mt * 4)?;
        let done = g.alloc(b * 4)?;
        let ctr = g.alloc(16)?;
        let x = h(&mut g, b * d)?;
        let hb = h(&mut g, b * d)?;
        let att = h(&mut g, b * d)?;
        let dout = h(&mut g, nl * b * (7 * d + ff))?;
        let dbias = h(&mut g, nl * b * (7 * d + ff))?;
        let logits = h(&mut g, b * dims.vocab)?;
        let kc = h(&mut g, nl * b * mt * d)?;
        let vc = h(&mut g, nl * b * mt * d)?;
        let s = g.alloc(b * nh * TENC * 4)?;
        let stat = g.alloc(b * nh * NCH * 8)?;
        let part = g.alloc(b * nh * NCH * HD * 4)?;
        let align = h(&mut g, b * mt * na * TENC)?;
        let e = Engine {
            g,
            dims,
            gc,
            enc,
            dec,
            conv1,
            conv2,
            epos,
            eln,
            xkv,
            emb,
            dpos,
            dln,
            bmax,
            eg,
            load_times,
            fused_attn: true,
            trace: None,
            mel,
            x1,
            h1,
            x2,
            ex,
            eh,
            eqkv,
            esc,
            eatt,
            ef,
            xkv_buf,
            tok,
            done,
            ctr,
            x,
            hb,
            att,
            dout,
            dbias,
            logits,
            kc,
            vc,
            s,
            stat,
            part,
            align,
            aslot,
            sup,
            sup_lang,
        };
        for (i, l) in e.dec.iter().enumerate() {
            let o = e.dec_out(i);
            for (y, lin, n) in [(o.qkv, &l.qkv, 3 * d), (o.o, &l.o, d), (o.xq, &l.xq, d), (o.xo, &l.xo, d), (o.fc1, &l.fc1, ff), (o.fc2, &l.fc2, d)] {
                let dst = e.dbias + (y - e.dout);
                e.g.k1("fill_bias", bmax * n / 8, &[A::P(dst), A::P(lin.b), A::L(bmax as i64), A::I(n as i32), A::L(n as i64)])?;
            }
        }
        e.g.sync()?;
        Ok(e)
    }

    /// Decoder layer i's output buffers inside dout.
    fn dec_out(&self, i: usize) -> DecOut {
        let (d, ff, b) = (self.dims.d, self.dims.ff, self.bmax);
        let base = self.dout + (2 * i * b * (7 * d + ff)) as u64;
        let at = |halves: usize| base + (2 * b * halves) as u64;
        DecOut { qkv: at(0), o: at(3 * d), xq: at(4 * d), xo: at(5 * d), fc1: at(6 * d), fc2: at(6 * d + ff) }
    }

    #[allow(clippy::too_many_arguments)]
    fn lin(&self, x: Ptr, ldx: usize, l: &Lin, y: Ptr, ldy: usize, m: usize, n: usize, k: usize) -> R<()> {
        self.g.gemm(x, ldx, l.w, y, ldy, m, n, k, Some(l.b))
    }

    fn ln(&self, x: Ptr, l: &Ln, y: Ptr, rows: usize) -> R<()> {
        self.g.k("layernorm", (rows as u32, 1, 1), 128, 0, &[A::P(x), A::P(l.w), A::P(l.b), A::P(y), A::I(self.dims.d as i32), A::F(EPS)])
    }

    /// x += dx, then y = layernorm(x); y may be dx.
    fn add_ln(&self, x: Ptr, dx: Ptr, l: &Ln, y: Ptr, rows: usize) -> R<()> {
        let a = [A::P(x), A::P(dx), A::P(l.w), A::P(l.b), A::P(y), A::I(self.dims.d as i32), A::F(EPS)];
        self.g.k("add_layernorm", (rows as u32, 1, 1), 128, 0, &a)
    }

    fn gelu(&self, x: Ptr, n: usize) -> R<()> {
        self.g.k1("gelu_inplace", n / 2, &[A::P(x), A::L((n / 2) as i64)])
    }

    /// Encoder for up to `eg` clips (log-mel [n_mels][3000] each); leaves the encoder output in `eh` and writes
    /// the cross-attention k/v of every decoder layer into decoder slots slot0.. .
    pub fn encode(&mut self, mels: &[&[f32]], slot0: usize) -> R<()> {
        let gn = mels.len();
        assert!(gn <= self.eg && slot0 + gn <= self.bmax);
        let (d, nm, nh) = (self.dims.d, self.dims.n_mels, self.dims.nh);
        let host: Vec<u16> = mels.iter().flat_map(|m| m.iter().map(|&v| f2h(v))).collect();
        self.g.upload(self.mel, &host)?;
        let (r1, r2) = (gn * NFRAMES, gn * TENC);
        self.g.k1("im2col1", r1 * 3 * nm, &[A::P(self.mel), A::P(self.x1), A::L((r1 * 3 * nm) as i64), A::I(nm as i32)])?;
        self.lin(self.x1, 3 * nm, &self.conv1, self.h1, d, r1, d, 3 * nm)?;
        self.gelu(self.h1, r1 * d)?;
        self.g.k1("im2col2", r2 * 3 * d, &[A::P(self.h1), A::P(self.x2), A::L((r2 * 3 * d) as i64), A::I(d as i32)])?;
        self.lin(self.x2, 3 * d, &self.conv2, self.ex, d, r2, d, 3 * d)?;
        self.gelu(self.ex, r2 * d)?;
        self.g.k1("add_pos", r2 * d, &[A::P(self.ex), A::P(self.epos), A::L(r2 as i64), A::I(d as i32), A::I(TENC as i32)])?;
        self.ln(self.ex, &self.enc[0].ln1, self.eh, r2)?;
        for (i, l) in self.enc.iter().enumerate() {
            if let Some(t) = self.trace.as_mut() {
                t.push(self.g.download_h(self.ex, TENC * d)?);
            }
            self.lin(self.eh, d, &l.qkv, self.eqkv, 3 * d, r2, 3 * d, d)?;
            if self.fused_attn {
                self.g.k("enc_flash", (TENC.div_ceil(64) as u32, nh as u32, gn as u32), 128, 0, &[A::P(self.eqkv), A::P(self.eatt), A::I(d as i32)])?;
            } else {
                for c in 0..gn {
                    let q = self.eqkv + (2 * c * TENC * 3 * d) as u64;
                    let (k, v) = (q + 2 * d as u64, q + 4 * d as u64);
                    // scores^T [key j, query i] per head = K Q^T, scaled by 1/8 (q * 0.125 is exact in fp16)
                    let mm = Mm {
                        ta: true,
                        tb: false,
                        m: TENC,
                        n: TENC,
                        k: HD,
                        lda: 3 * d,
                        ldb: 3 * d,
                        ldc: TENC,
                        sa: HD as i64,
                        sb: HD as i64,
                        sc: (TENC * TENC) as i64,
                        batch: nh,
                    };
                    self.g.matmul(&mm, 0.125, k, q, 0.0, self.esc)?;
                    self.g.k("softmax_rows", ((nh * TENC) as u32, 1, 1), 256, 0, &[A::P(self.esc), A::I(TENC as i32)])?;
                    let out = self.eatt + (2 * c * TENC * d) as u64;
                    let mm = Mm {
                        ta: false,
                        tb: false,
                        m: HD,
                        n: TENC,
                        k: TENC,
                        lda: 3 * d,
                        ldb: TENC,
                        ldc: d,
                        sa: HD as i64,
                        sb: (TENC * TENC) as i64,
                        sc: HD as i64,
                        batch: nh,
                    };
                    self.g.matmul(&mm, 1.0, v, self.esc, 0.0, out)?;
                }
            }
            self.lin(self.eatt, d, &l.o, self.eh, d, r2, d, d)?;
            self.add_ln(self.ex, self.eh, &l.ln2, self.eh, r2)?;
            self.lin(self.eh, d, &l.fc1, self.ef, self.dims.ff, r2, self.dims.ff, d)?;
            self.gelu(self.ef, r2 * self.dims.ff)?;
            self.lin(self.ef, self.dims.ff, &l.fc2, self.eh, d, r2, d, self.dims.ff)?;
            let next = self.enc.get(i + 1).map(|n| &n.ln1).unwrap_or(&self.eln); // the last one is the final layer norm
            self.add_ln(self.ex, self.eh, next, self.eh, r2)?;
        }
        // cross k/v per (layer, k|v, head) as [position, HD] blocks: one batched GEMM per clip, HD output rows each,
        // accumulated onto the bias
        let xkv = self.dims.dec_layers * 2 * d * TENC;
        let blocks = self.dims.dec_layers * 2 * nh;
        let mm = Mm { ta: true, tb: false, m: HD, n: TENC, k: d, lda: d, ldb: d, ldc: HD, sa: (HD * d) as i64, sb: 0, sc: (TENC * HD) as i64, batch: blocks };
        for c in 0..gn {
            let dst = self.xkv_buf + (2 * (slot0 + c) * xkv) as u64;
            self.g.k1("fill_bias_heads", xkv / 8, &[A::P(dst), A::P(self.xkv.b), A::L((xkv / 8) as i64)])?;
            self.g.matmul(&mm, 1.0, self.xkv.w, self.eh + (2 * c * TENC * d) as u64, 1.0, dst)?;
        }
        Ok(())
    }

    /// Prompt tokens into each slot, all sequences unfinished, position 0.
    pub fn reset(&mut self, prompts: &[Vec<i32>]) -> R<()> {
        let mt = self.dims.max_t;
        let mut t = vec![0i32; prompts.len() * mt];
        for (s, p) in prompts.iter().enumerate() {
            t[s * mt..s * mt + p.len()].copy_from_slice(p);
        }
        self.g.upload(self.tok, &t)?;
        self.g.upload(self.done, &vec![0i32; prompts.len()])?;
        self.g.upload(self.ctr, &[0i32; 4])
    }

    /// Language detection: one step on the start token with every non-language token suppressed. Writes the
    /// detected language token at position 1 of each slot (the rest of the prompt stays) and returns them.
    pub fn detect(&self, b: usize) -> R<Vec<i32>> {
        self.step(b, true, 1)?;
        let mt = self.dims.max_t;
        let t = self.g.download::<i32>(self.tok, b * mt)?;
        Ok((0..b).map(|s| t[s * mt + 1]).collect())
    }

    /// One decoding step for b sequences: feed token p, choose token p + 1 (positions below prompt_len stay
    /// as they are), advance p. `lang` restricts the choice to language tokens.
    pub fn step(&self, b: usize, lang: bool, prompt_len: usize) -> R<()> {
        let g = &self.g;
        let (d, ff, nh, nl, mt) = (self.dims.d, self.dims.ff, self.dims.nh, self.dims.dec_layers, self.dims.max_t);
        let (di, mti, nli) = (d as i32, mt as i32, nl as i32);
        let na = self.gc.align_heads.len() as i32;
        g.k("embed", (b as u32, 1, 1), 256, 0, &[A::P(self.tok), A::P(self.ctr), A::P(self.emb), A::P(self.dpos), A::P(self.x), A::I(mti), A::I(di)])?;
        self.ln(self.x, &self.dec[0].ln1, self.hb, b)?;
        g.copy(self.dout, self.dbias, 2 * nl * self.bmax * (7 * d + ff))?;
        // y (holding the bias) += x W^T
        let lin = |x: Ptr, ldx: usize, l: &Lin, y: Ptr, n: usize, k: usize| g.gemm_onto_bias(x, ldx, l.w, y, n, b, n, k, l.b);
        for (i, l) in self.dec.iter().enumerate() {
            let o = self.dec_out(i);
            let kc = self.kc + (2 * i * self.bmax * mt * d) as u64;
            let vc = self.vc + (2 * i * self.bmax * mt * d) as u64;
            lin(self.hb, d, &l.qkv, o.qkv, 3 * d, d)?;
            g.k("kv_write", (b as u32, 1, 1), 256, 0, &[A::P(o.qkv), A::P(kc), A::P(vc), A::P(self.ctr), A::I(mti), A::I(di)])?;
            g.k(
                "self_attn",
                (b as u32, nh as u32, 1),
                128,
                0,
                &[A::P(o.qkv), A::P(kc), A::P(vc), A::P(self.att), A::P(self.ctr), A::P(self.done), A::I(mti), A::I(di)],
            )?;
            lin(self.att, d, &l.o, o.o, d, d)?;
            self.add_ln(self.x, o.o, &l.lnx, self.hb, b)?;
            lin(self.hb, d, &l.xq, o.xq, d, d)?;
            let grid = ((b * nh) as u32, NCH as u32, 1);
            g.k(
                "cross_scores",
                grid,
                128,
                0,
                &[A::P(o.xq), A::P(self.xkv_buf), A::P(self.s), A::P(self.stat), A::P(self.done), A::I(i as i32), A::I(nli), A::I(di)],
            )?;
            g.k(
                "cross_pv",
                grid,
                128,
                0,
                &[
                    A::P(self.s),
                    A::P(self.stat),
                    A::P(self.xkv_buf),
                    A::P(self.part),
                    A::P(self.align),
                    A::P(self.aslot),
                    A::P(self.ctr),
                    A::P(self.done),
                    A::I(i as i32),
                    A::I(nli),
                    A::I(di),
                    A::I(na),
                    A::I(mti),
                ],
            )?;
            g.k("cross_combine", ((b * nh) as u32, 1, 1), HD as u32, 0, &[A::P(self.part), A::P(self.att), A::P(self.done), A::I(di)])?;
            lin(self.att, d, &l.xo, o.xo, d, d)?;
            self.add_ln(self.x, o.xo, &l.ln3, self.hb, b)?;
            lin(self.hb, d, &l.fc1, o.fc1, ff, d)?;
            self.gelu(o.fc1, b * ff)?;
            lin(o.fc1, ff, &l.fc2, o.fc2, d, ff)?;
            let next = self.dec.get(i + 1).map(|n| &n.ln1).unwrap_or(&self.dln);
            self.add_ln(self.x, o.fc2, next, self.hb, b)?;
        }
        g.gemm(self.hb, d, self.emb, self.logits, self.dims.vocab, b, self.dims.vocab, d, None)?;
        let bs = |i: usize| if lang { -1 } else { self.gc.begin_suppress.get(i).copied().unwrap_or(-1) };
        g.k(
            "argmax_tok",
            (b as u32, 1, 1),
            1024,
            0,
            &[
                A::P(self.logits),
                A::I(self.dims.vocab as i32),
                A::P(if lang { self.sup_lang } else { self.sup }),
                A::P(self.tok),
                A::P(self.done),
                A::P(self.ctr),
                A::I(mti),
                A::I(prompt_len as i32),
                A::I(self.gc.eos),
                A::I(bs(0)),
                A::I(bs(1)),
            ],
        )?;
        g.k("advance", (1, 1, 1), 1, 0, &[A::P(self.ctr)])
    }
}
