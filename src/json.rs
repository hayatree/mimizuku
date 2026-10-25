//! A strict JSON parser (RFC 8259) for safetensors headers, configs and vocabularies.
type R<T> = Result<T, Box<dyn std::error::Error>>;

const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone)]
pub enum Json {
    Obj(Vec<(String, Json)>),
    Arr(Vec<Json>),
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(n, _)| n == k).map(|(_, j)| j),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<&str> {
        if let Json::Str(s) = self { Some(s) } else { None }
    }
    pub fn num(&self) -> Option<f64> {
        if let Json::Num(n) = self { Some(*n) } else { None }
    }
    /// A non-negative integer that f64 represents exactly.
    pub fn uint(&self) -> Option<usize> {
        self.num().filter(|n| *n >= 0.0 && n.fract() == 0.0 && *n <= 9_007_199_254_740_992.0).map(|n| n as usize)
    }
    /// An array of non-negative integers; None if any entry is not one.
    pub fn uints(&self) -> Option<Vec<usize>> {
        if let Json::Arr(v) = self { v.iter().map(|j| j.uint()).collect() } else { None }
    }
    pub fn bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self { Some(*b) } else { None }
    }
}

struct P<'a> {
    s: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> R<u8> {
        Ok(*self.s.get(self.i).ok_or("json: unexpected end")?)
    }
    fn next(&mut self) -> R<u8> {
        let c = self.peek()?;
        self.i += 1;
        Ok(c)
    }
    fn expect(&mut self, c: u8) -> R<()> {
        self.ws();
        if self.next()? != c {
            return Err(format!("json: expected '{}' at byte {}", c as char, self.i - 1).into());
        }
        Ok(())
    }
    fn hex4(&mut self) -> R<u32> {
        let h = self.s.get(self.i..self.i + 4).filter(|h| h.iter().all(u8::is_ascii_hexdigit)).ok_or("json: bad \\u escape")?;
        self.i += 4;
        Ok(u32::from_str_radix(std::str::from_utf8(h)?, 16)?)
    }
    fn string(&mut self) -> R<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            match self.next()? {
                b'"' => break,
                b'\\' => match self.next()? {
                    b'n' => out.push(b'\n'),
                    b't' => out.push(b'\t'),
                    b'r' => out.push(b'\r'),
                    b'b' => out.push(8),
                    b'f' => out.push(12),
                    b'u' => {
                        let mut cp = self.hex4()?;
                        // a high surrogate pairs with an immediately following low one; unpaired ones become U+FFFD
                        if (0xd800..0xdc00).contains(&cp) && self.s.get(self.i..self.i + 2) == Some(b"\\u") {
                            let save = self.i;
                            self.i += 2;
                            let lo = self.hex4()?;
                            if (0xdc00..0xe000).contains(&lo) {
                                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            } else {
                                self.i = save;
                            }
                        }
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(char::from_u32(cp).unwrap_or('\u{fffd}').encode_utf8(&mut buf).as_bytes());
                    }
                    c @ (b'"' | b'\\' | b'/') => out.push(c),
                    c => return Err(format!("json: bad escape \\{} at byte {}", c as char, self.i - 1).into()),
                },
                c if c < 0x20 => return Err(format!("json: control character in a string at byte {}", self.i - 1).into()),
                c => out.push(c),
            }
        }
        Ok(String::from_utf8(out)?)
    }
    /// -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?, finite
    fn number(&mut self) -> R<Json> {
        let st = self.i;
        let digits = |p: &mut Self| {
            let d = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - d
        };
        if self.s.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let lead = self.s.get(self.i).copied();
        let n = digits(self);
        let mut ok = n == 1 || (n > 1 && lead != Some(b'0'));
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            ok &= digits(self) > 0;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            ok &= digits(self) > 0;
        }
        let v: f64 = std::str::from_utf8(&self.s[st..self.i])?.parse().unwrap_or(f64::NAN);
        if !ok || !v.is_finite() {
            return Err(format!("json: bad number at byte {st}").into());
        }
        Ok(Json::Num(v))
    }
    fn val(&mut self, depth: usize) -> R<Json> {
        if depth > MAX_DEPTH {
            return Err("json: nested too deeply".into());
        }
        self.ws();
        match self.peek()? {
            b'{' => {
                self.i += 1;
                let mut v = vec![];
                self.ws();
                if self.peek()? == b'}' {
                    self.i += 1;
                    return Ok(Json::Obj(v));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.expect(b':')?;
                    v.push((k, self.val(depth + 1)?));
                    self.ws();
                    match self.next()? {
                        b',' => continue,
                        b'}' => break,
                        _ => return Err(format!("json: expected ',' or '}}' at byte {}", self.i - 1).into()),
                    }
                }
                Ok(Json::Obj(v))
            }
            b'[' => {
                self.i += 1;
                let mut v = vec![];
                self.ws();
                if self.peek()? == b']' {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                loop {
                    v.push(self.val(depth + 1)?);
                    self.ws();
                    match self.next()? {
                        b',' => continue,
                        b']' => break,
                        _ => return Err(format!("json: expected ',' or ']' at byte {}", self.i - 1).into()),
                    }
                }
                Ok(Json::Arr(v))
            }
            b'"' => Ok(Json::Str(self.string()?)),
            b't' | b'f' | b'n' => {
                let st = self.i;
                while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() {
                    self.i += 1;
                }
                match &self.s[st..self.i] {
                    b"true" => Ok(Json::Bool(true)),
                    b"false" => Ok(Json::Bool(false)),
                    b"null" => Ok(Json::Null),
                    _ => Err(format!("json: bad literal at byte {st}").into()),
                }
            }
            _ => self.number(),
        }
    }
}

pub fn parse_json(s: &[u8]) -> R<Json> {
    let mut p = P { s, i: 0 };
    let v = p.val(0)?;
    p.ws();
    if p.i != s.len() {
        return Err(format!("json: trailing data at byte {}", p.i).into());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_values() {
        let j = parse_json(br#" {"a": [1, 2.5, -3e2], "b": "x\"\u00e9\ud83d\ude00\r", "c": true, "d": null, "e": {}} "#).unwrap();
        let Some(Json::Arr(a)) = j.get("a") else { panic!() };
        assert_eq!(a.iter().map(|x| x.num()).collect::<Vec<_>>(), [Some(1.0), Some(2.5), Some(-300.0)]);
        assert_eq!(j.get("a").unwrap().uints(), None);
        assert_eq!(parse_json(b"[0, 7]").unwrap().uints(), Some(vec![0, 7]));
        assert_eq!(j.get("b").unwrap().str(), Some("x\"é😀\r"));
        assert_eq!(j.get("c").unwrap().bool(), Some(true));
        assert!(matches!(j.get("d"), Some(Json::Null)));
        assert!(matches!(j.get("e"), Some(Json::Obj(v)) if v.is_empty()));
    }

    #[test]
    fn json_malformed_is_an_error() {
        for s in [
            &b"{"[..],
            b"[1,",
            b"{\"a\" 1}",
            b"\"abc",
            b"[1] x",
            b"tru",
            b"\"\\u12\"",
            b"-",
            b"",
            b"01",
            b"1.",
            b".5",
            b"1e",
            b"+1",
            b"1e999",
            b"\"\\q\"",
            b"\"a\tb\"",
            b"\"\\u+041\"",
        ] {
            assert!(parse_json(s).is_err(), "{:?}", String::from_utf8_lossy(s));
        }
        let deep = "[".repeat(1000) + &"]".repeat(1000);
        assert!(parse_json(deep.as_bytes()).is_err());
    }

    #[test]
    fn json_unpaired_surrogate_keeps_what_follows() {
        assert_eq!(parse_json(br#""\ud800\u0041""#).unwrap().str(), Some("\u{fffd}A"));
    }
}
