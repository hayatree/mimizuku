//! Safetensors reader: header validation, memory mapping, typed tensor access.
use crate::json::{Json, parse_json};
use std::collections::HashMap;

type R<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Tensor<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// File contents: memory-mapped where possible (no copy of a multi-GB checkpoint), read into memory otherwise.
enum Buf {
    Owned(Vec<u8>),
    #[cfg(unix)]
    Mapped(map::Map),
}

impl Buf {
    fn bytes(&self) -> &[u8] {
        match self {
            Buf::Owned(v) => v,
            #[cfg(unix)]
            Buf::Mapped(m) => m.bytes(),
        }
    }
}

#[cfg(unix)]
mod map {
    use std::ffi::c_void;
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;
    const MADV_WILLNEED: i32 = 3;

    /// A read-only private mapping of a whole file. The file must not be truncated while it is mapped.
    pub struct Map {
        ptr: *mut c_void,
        len: usize,
    }

    impl Map {
        pub fn new(f: &std::fs::File) -> Option<Map> {
            let len = f.metadata().ok()?.len() as usize;
            if len == 0 {
                return None;
            }
            let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, f.as_raw_fd(), 0) };
            if ptr as isize == -1 {
                return None;
            }
            unsafe { madvise(ptr, len, MADV_WILLNEED) }; // start reading ahead; a failure only costs speed
            Some(Map { ptr, len })
        }
        pub fn bytes(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            unsafe { munmap(self.ptr, self.len) };
        }
    }
}

/// A whole safetensors file.
pub struct File {
    buf: Buf,
    start: usize,
    index: HashMap<String, (String, Vec<usize>, usize, usize)>,
    pub meta: HashMap<String, String>,
}

fn elem_size(dt: &str) -> Option<usize> {
    Some(match dt {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}

impl File {
    pub fn open(path: &str) -> R<Self> {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        #[cfg(unix)]
        let buf = match map::Map::new(&file) {
            Some(m) => Buf::Mapped(m),
            None => Buf::Owned(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?),
        };
        #[cfg(not(unix))]
        let buf = Buf::Owned(std::fs::read(path).map_err(|e| format!("{path}: {e}"))?);
        drop(file);
        let bytes = buf.bytes();
        let bad = |what: &str| format!("{path}: not a valid safetensors file ({what})");
        let n = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| bad("too short"))?.try_into()?) as usize;
        let head = bytes.get(8..8usize.saturating_add(n)).ok_or_else(|| bad("header length"))?;
        let Json::Obj(items) = parse_json(head)? else { return Err(bad("header is not an object").into()) };
        let (start, data_len) = (8 + n, bytes.len() - 8 - n);
        let mut index = HashMap::new();
        let mut meta = HashMap::new();
        let mut ranges = vec![];
        for (k, v) in items {
            if k == "__metadata__" {
                if let Json::Obj(m) = v {
                    for (mk, mv) in m {
                        meta.insert(mk, mv.str().unwrap_or("").to_string());
                    }
                }
                continue;
            }
            let dt = v.get("dtype").and_then(|j| j.str()).ok_or_else(|| bad("dtype"))?.to_string();
            let shape = v.get("shape").and_then(|j| j.uints()).ok_or_else(|| bad(&format!("shape of {k}")))?;
            let off = v.get("data_offsets").and_then(|j| j.uints()).unwrap_or_default();
            let numel = shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d));
            let size = elem_size(&dt).zip(numel).and_then(|(e, n)| e.checked_mul(n));
            if off.len() != 2 || off[0] > off[1] || off[1] > data_len || size != Some(off[1] - off[0]) {
                return Err(bad(&format!("offsets of {k}")).into());
            }
            ranges.push((off[0], off[1]));
            if index.insert(k.clone(), (dt, shape, off[0], off[1])).is_some() {
                return Err(bad(&format!("{k} appears twice")).into());
            }
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(bad("overlapping tensors").into());
        }
        Ok(File { buf, start, index, meta })
    }

    pub fn get(&self, name: &str) -> R<Tensor<'_>> {
        let (dt, shape, a, b) = self.index.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
        Ok(Tensor { dtype: dt.clone(), shape: shape.clone(), data: &self.buf.bytes()[self.start + a..self.start + b] })
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn tensor_count(&self) -> usize {
        self.index.len()
    }

    /// Bytes all tensors take as fp16.
    pub fn f16_bytes(&self) -> usize {
        self.index.values().map(|(dt, _, a, b)| (b - a) / elem_size(dt).unwrap_or(2) * 2).sum()
    }

    pub fn f32s(&self, name: &str) -> R<(Vec<f32>, Vec<usize>)> {
        let t = self.get(name)?;
        if t.dtype != "F32" {
            return Err(format!("{name}: expected F32, got {}", t.dtype).into());
        }
        Ok((t.data.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(), t.shape))
    }

    /// F16 or F32 tensor as f32.
    pub fn floats(&self, name: &str) -> R<(Vec<f32>, Vec<usize>)> {
        let t = self.get(name)?;
        if t.dtype == "F16" {
            return Ok((t.data.as_chunks::<2>().0.iter().map(|&c| crate::gpu::h2f(u16::from_le_bytes(c))).collect(), t.shape));
        }
        self.f32s(name)
    }

    pub fn i32s(&self, name: &str) -> R<(Vec<i32>, Vec<usize>)> {
        let t = self.get(name)?;
        if t.dtype != "I32" {
            return Err(format!("{name}: expected I32, got {}", t.dtype).into());
        }
        Ok((t.data.as_chunks::<4>().0.iter().map(|&c| i32::from_le_bytes(c)).collect(), t.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st_file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(header.as_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn safetensors_checks_offsets() {
        let dir = std::env::temp_dir().join(format!("we-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let ok = st_file(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#, &[0, 0, 128, 63, 0, 0, 0, 64]);
        std::fs::write(&p, ok).unwrap();
        let f = File::open(p.to_str().unwrap()).unwrap();
        assert_eq!(f.f32s("x").unwrap().0, vec![1.0, 2.0]);
        for bad in [
            st_file(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,16]}}"#, &[0; 8]),
            st_file(r#"{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#, &[0; 8]),
            st_file(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#, &[0; 8]),
            st_file(r#"{"x":{"dtype":"F32","shape":[2,"a"],"data_offsets":[0,8]}}"#, &[0; 8]),
            st_file(r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#, &[0; 8]),
            vec![255; 12],
            vec![1, 2],
        ] {
            std::fs::write(&p, bad).unwrap();
            assert!(File::open(p.to_str().unwrap()).is_err());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
