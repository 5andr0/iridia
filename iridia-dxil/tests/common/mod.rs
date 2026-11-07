#![allow(dead_code)]
//! What the corpus tests share: a game's shader corpus, read through its tree (a folder or an archive).

use std::sync::Mutex;

use crate::inputs::{Game, OracleSet, Query, expect_found};
use primase_testdata::Tree;

/// A game's shader corpus (`oracles.corpus.<game>/shaders`: a folder, searched recursively, or an archive of
/// containers as the game ships them); `None` when it is not configured. A configured corpus without a container fails.
pub fn corpus(game: Game) -> Option<Corpus> {
    let query = Query::Oracle(OracleSet::Corpus, game, "shaders");
    let tree = crate::inputs::oracle_tree(OracleSet::Corpus, game, "shaders")?;
    let corpus = Corpus::of(query.key(), tree);
    expect_found(query, "DXBC containers", corpus.names.len());
    Some(corpus)
}

/// The DXBC containers of a [`Tree`] (a folder or an archive): every file whose bytes begin with `DXBC`, in path order,
/// as a walk of the folder visits them (each folder's entries sorted, a folder's files where its name sorts).
pub struct Corpus {
    /// What reports call it: the corpus's key, or the folder.
    pub name: String,
    /// The containers' names in the tree.
    pub names: Vec<String>,
    tree: Tree,
    /// Held while a file of a folder is read, so one is read at a time: files of the NAS's SMB share opened at once on
    /// several threads fail at random with Windows error 1450 (`ERROR_NO_SYSTEM_RESOURCES`). An archive is read
    /// from one open file at positions, on as many threads as read it.
    folder_reads: Option<Mutex<()>>,
}

impl Corpus {
    /// The containers of `tree`, called `name` in reports.
    pub fn of(name: String, tree: Tree) -> Self {
        let folder_reads = tree.as_folder().map(|_| Mutex::new(()));
        let mut corpus = Corpus {
            name,
            names: Vec::new(),
            tree,
            folder_reads,
        };
        let mut names: Vec<&str> = corpus.tree.names();
        names.sort_by(|a, b| a.split('/').cmp(b.split('/')));
        let names: Vec<String> = names
            .into_iter()
            .filter(|n| corpus.read(n).starts_with(b"DXBC"))
            .map(str::to_owned)
            .collect();
        corpus.names = names;
        corpus
    }

    /// The bytes of the container `name`.
    pub fn read(&self, name: &str) -> Vec<u8> {
        let _one_at_a_time = self.folder_reads.as_ref().map(|m| m.lock().unwrap());
        self.tree
            .read(name)
            .unwrap_or_else(|e| panic!("{}: {e}", self.at(name)))
    }

    /// Where the container `name` is, for a report: the folder or archive and the name in it.
    pub fn at(&self, name: &str) -> String {
        format!("{}/{name}", self.tree.path().display())
    }
}
