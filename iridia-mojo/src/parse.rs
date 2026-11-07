//! What the lowerings of the dialect share: a cursor over its tokens, and the facts of the dialect that do not depend
//! on what a program is lowered to (the varyings' semantics, the swizzles, the targets a pixel program writes and the
//! clip-space fix MojoShader ends a vertex program with).

use std::collections::BTreeSet;

use crate::lex::{Spanned, Token};
use crate::{Result, error};

/// The tokens of a program and the place being read.
pub(crate) struct Cursor {
    pub tokens: Vec<Spanned>,
    pub pos: usize,
}

impl Cursor {
    pub(crate) fn new(tokens: Vec<Spanned>) -> Cursor {
        Cursor { tokens, pos: 0 }
    }

    pub(crate) fn line(&self) -> usize {
        self.tokens
            .get(self.pos)
            .or_else(|| self.tokens.last())
            .map_or(0, |t| t.line)
    }

    pub(crate) fn fail<T>(&self, detail: impl Into<String>) -> Result<T> {
        Err(error(self.line(), detail))
    }

    pub(crate) fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    pub(crate) fn next(&mut self) -> Result<Token> {
        match self.tokens.get(self.pos) {
            Some(t) => {
                self.pos = self.pos.saturating_add(1);
                Ok(t.token.clone())
            }
            None => self.fail("the program ends in the middle of a statement"),
        }
    }

    pub(crate) fn at(&self, punct: &str) -> bool {
        self.peek().is_some_and(|t| t.is(punct))
    }

    pub(crate) fn at_ident(&self, word: &str) -> bool {
        matches!(self.peek(), Some(Token::Ident(s)) if s == word)
    }

    pub(crate) fn accept(&mut self, punct: &str) -> bool {
        let found = self.at(punct);
        if found {
            self.pos = self.pos.saturating_add(1);
        }
        found
    }

    pub(crate) fn expect(&mut self, punct: &str) -> Result<()> {
        if self.accept(punct) {
            Ok(())
        } else {
            let found = self.peek().map_or("the end".to_owned(), Token::text);
            self.fail(format!("{punct} expected, found {found}"))
        }
    }

    pub(crate) fn ident(&mut self) -> Result<String> {
        match self.next()? {
            Token::Ident(s) => Ok(s),
            other => self.fail(format!("a name expected, found {}", other.text())),
        }
    }

    pub(crate) fn word(&mut self, word: &str) -> Result<()> {
        let found = self.ident()?;
        if found == word {
            Ok(())
        } else {
            self.fail(format!("{word} expected, found {found}"))
        }
    }

    pub(crate) fn integer(&mut self) -> Result<i64> {
        match self.next()? {
            Token::Int(i) => Ok(i),
            other => self.fail(format!("an integer expected, found {}", other.text())),
        }
    }

    pub(crate) fn has_ident(&self, word: &str) -> bool {
        self.tokens
            .iter()
            .any(|t| matches!(&t.token, Token::Ident(s) if s == word))
    }

    /// The targets a pixel program writes, which it never declares: `gl_FragColor` is target 0, `gl_FragData[n]`
    /// target `n` (0 to 3).
    pub(crate) fn fragment_targets(&self) -> Result<BTreeSet<u32>> {
        let mut targets = BTreeSet::new();
        for (i, t) in self.tokens.iter().enumerate() {
            match &t.token {
                Token::Ident(s) if s == "gl_FragColor" => {
                    targets.insert(0u32);
                }
                Token::Ident(s) if s == "gl_FragData" => {
                    let n = match (
                        self.tokens.get(i.saturating_add(1)).map(|t| &t.token),
                        self.tokens.get(i.saturating_add(2)).map(|t| &t.token),
                    ) {
                        (Some(open), Some(Token::Int(n))) if open.is("[") => *n,
                        _ => {
                            return Err(error(
                                t.line,
                                "gl_FragData indexed by something that is not a number",
                            ));
                        }
                    };
                    let n = u32::try_from(n)
                        .ok()
                        .filter(|n| *n < 4)
                        .ok_or_else(|| error(t.line, format!("gl_FragData[{n}]")))?;
                    targets.insert(n);
                }
                _ => {}
            }
        }
        Ok(targets)
    }

    /// Read `void main() {` and check that the program ends with main's closing brace; of a vertex program that reads
    /// `posFixup`, drop the clip-space fix MojoShader ends it with, and say so. Gives the note, and leaves the cursor at
    /// main's first statement.
    pub(crate) fn open_main(&mut self, vertex: bool) -> Result<Option<String>> {
        self.word("void")?;
        self.word("main")?;
        self.expect("(")?;
        self.expect(")")?;
        self.expect("{")?;
        // the end of the body: its closing brace is the last token
        if self.tokens.last().is_none_or(|t| !t.token.is("}")) {
            return self.fail("main does not end with a closing brace");
        }
        if !(vertex && self.has_ident("posFixup")) {
            return Ok(None);
        }
        let end = self.tokens.len().saturating_sub(1);
        let tail: Vec<String> = self.tokens.iter().take(end).map(|t| t.token.text()).collect();
        let epilogue = EPILOGUE.join(" ");
        if !tail.join(" ").ends_with(&epilogue) {
            return self
                .fail("posFixup used outside the clip-space fix MojoShader ends a vertex program with");
        }
        let matched: usize = EPILOGUE.iter().map(|s| s.split(' ').count()).sum();
        self.tokens.drain(end.saturating_sub(matched)..end);
        Ok(Some(
            "the clip-space fix for OpenGL that MojoShader ends a vertex program with (posFixup: flip y, half-pixel \
             offset, depth 0..1 to -1..1) is dropped: the program keeps Direct3D's clip space"
                .into(),
        ))
    }

    /// Whether main's statements are all read: its closing brace, the last token, is next.
    pub(crate) fn main_done(&self) -> bool {
        self.pos >= self.tokens.len().saturating_sub(1)
    }
}

/// The clip-space fix MojoShader ends every vertex program with, as tokens joined by spaces.
const EPILOGUE: [&str; 3] = [
    "gl_Position . y = gl_Position . y * posFixup . y ;",
    "gl_Position . xy += posFixup . zw * gl_Position . ww ;",
    "gl_Position . z = gl_Position . z * 2.0 - gl_Position . w ;",
];

/// The semantic a varying of the dialect stands for, as MojoShader's GLSL profile names them: `vFrontColor` is colour 0,
/// `vFrontSecondaryColor` colour 1, `vFogFragCoord` the fog, `vTexCoord<n>` texture coordinate n.
pub(crate) fn varying_semantic(name: &str) -> Option<(&'static str, u32)> {
    match name {
        "vFrontColor" => Some(("COLOR", 0)),
        "vFrontSecondaryColor" => Some(("COLOR", 1)),
        "vFogFragCoord" => Some(("FOG", 0)),
        _ => {
            let n = name.strip_prefix("vTexCoord")?;
            Some(("TEXCOORD", n.parse().ok()?))
        }
    }
}

/// The lanes a swizzle names (`xyzw`, `rgba`, `stpq`), none when it is not one.
pub(crate) fn swizzle_lanes(word: &str) -> Option<Vec<u8>> {
    if word.is_empty() || word.len() > 4 {
        return None;
    }
    word.chars()
        .map(|c| match c {
            'x' | 'r' | 's' => Some(0),
            'y' | 'g' | 't' => Some(1),
            'z' | 'b' | 'p' => Some(2),
            'w' | 'a' | 'q' => Some(3),
            _ => None,
        })
        .collect()
}
