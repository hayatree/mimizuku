//! Decoding side of Whisper's byte-level BPE tokenizer (vocab.json), enough to turn generated ids into text.
use crate::gpu::R;
use crate::json::{Json, parse_json};

pub struct Tok {
    bytes: Vec<Vec<u8>>, // id -> raw bytes
}

/// GPT-2's byte <-> printable character table, inverted: char -> byte.
fn unicode_to_byte() -> std::collections::HashMap<char, u8> {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xa1..=0xac).chain(0xae..=0xff).collect();
    let mut chars: Vec<u32> = printable.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    printable.iter().zip(&chars).map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

impl Tok {
    /// `vocab`: the model's vocabulary size; `eos`: the first special token (every id below it is ordinary text).
    pub fn load(model_dir: &str, vocab: usize, eos: i32) -> R<Self> {
        let v = parse_json(&std::fs::read(format!("{model_dir}/vocab.json"))?)?;
        let map = unicode_to_byte();
        let Json::Obj(items) = v else { return Err("vocab.json: not an object".into()) };
        let mut bytes: Vec<Option<Vec<u8>>> = vec![None; vocab];
        for (s, id) in items {
            let slot = id.uint().and_then(|i| bytes.get_mut(i)).ok_or_else(|| format!("vocab.json: id of {s:?} is not below the model's {vocab}"))?;
            if slot.is_some() {
                return Err(format!("vocab.json: id of {s:?} is used twice").into());
            }
            *slot = Some(s.chars().map(|c| map.get(&c).copied().ok_or_else(|| format!("vocab.json: char {c:?}"))).collect::<Result<_, _>>()?);
        }
        if let Some(i) = bytes.iter().take(eos.max(0) as usize).position(Option::is_none) {
            return Err(format!("vocab.json: no entry for text token {i}").into());
        }
        Ok(Tok { bytes: bytes.into_iter().map(Option::unwrap_or_default).collect() })
    }

    /// Text of ordinary (non-special) ids as the HF tokenizer's decode gives it: invalid UTF-8 becomes U+FFFD,
    /// and clean_up_tokenization_spaces removes the space before punctuation and English contractions.
    pub fn decode(&self, ids: &[i32]) -> String {
        let b: Vec<u8> = ids.iter().flat_map(|&i| self.raw(i)).copied().collect();
        let mut s = String::from_utf8_lossy(&b).into_owned();
        for (a, r) in
            [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]
        {
            s = s.replace(a, r);
        }
        s
    }

    fn raw(&self, id: i32) -> &[u8] {
        usize::try_from(id).ok().and_then(|i| self.bytes.get(i)).map_or(&[], |b| b.as_slice())
    }

    /// Groups generated tokens into whole UTF-8 pieces with their start and end times: a piece starts at its first
    /// token's time and ends where the next token starts. Special and timestamp tokens (ids >= eos) are skipped.
    pub fn pieces(&self, ids: &[i32], ts: &[f32], eos: i32) -> Vec<(String, f32, f32)> {
        let (mut out, mut buf, mut bytes, mut t0) = (vec![], vec![], vec![], 0f32);
        for (i, (&id, &t)) in ids.iter().zip(ts).enumerate() {
            if id >= eos {
                continue;
            }
            if buf.is_empty() {
                t0 = t;
            }
            buf.push(id);
            bytes.extend_from_slice(self.raw(id));
            // a piece is complete once its bytes are valid UTF-8, or once they can no longer become valid
            if std::str::from_utf8(&bytes).map_or_else(|e| e.error_len().is_some(), |_| true) {
                out.push((self.decode(&buf), t0, ts.get(i + 1).copied().unwrap_or(t)));
                buf.clear();
                bytes.clear();
            }
        }
        if !buf.is_empty() {
            out.push((self.decode(&buf), t0, ts.last().copied().unwrap_or(t0)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_is_a_bijection() {
        let m = unicode_to_byte();
        assert_eq!(m.len(), 256);
        assert_eq!(m[&'Ġ'], b' ');
        assert_eq!(m[&'!'], b'!');
        let mut bytes: Vec<u8> = m.values().copied().collect();
        bytes.sort();
        assert_eq!(bytes, (0..=255).collect::<Vec<u8>>());
    }

    #[test]
    fn decode_joins_bytes_and_cleans_spaces() {
        // "日" is e6 97 a5: split over two tokens, as byte-level BPE does
        let t = Tok { bytes: vec![b" Hello".to_vec(), b" .".to_vec(), vec![0xe6, 0x97], vec![0xa5], b" n't".to_vec()] };
        assert_eq!(t.decode(&[0, 1]), " Hello.");
        assert_eq!(t.decode(&[2]), "\u{fffd}"); // incomplete UTF-8 so far
        assert_eq!(t.decode(&[2, 3]), "日");
        assert_eq!(t.decode(&[0, 4]), " Hellon't");
        assert_eq!(t.decode(&[99]), ""); // unknown ids are dropped
        assert_eq!(t.decode(&[-1]), "");
    }

    #[test]
    fn pieces_are_whole_characters() {
        let t = Tok { bytes: vec![b" a".to_vec(), vec![0xe6, 0x97], vec![0xa5], vec![0xff], vec![0xe6]] };
        let ts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let p = t.pieces(&[0, 1, 2, 3, 9, 4], &ts, 5);
        let want = [(" a", 0.0, 0.1), ("日", 0.1, 0.3), ("\u{fffd}", 0.3, 0.4), ("\u{fffd}", 0.5, 0.5)];
        assert_eq!(p, want.map(|(s, a, b)| (s.to_string(), a, b)));
    }
}
