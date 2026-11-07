#![allow(clippy::unwrap_used)]
//! The container digest over the games' own Shader Model 6 containers: for every container of a corpus,
//! `dxcontainer::checksum` equals the digest the compiler stored, the stored digest is not all zero (so no stock
//! container is unsigned), the part directory is read to the exact end (parts of Elden Ring's are not padded to four
//! bytes), and `dxcontainer::serialize` writes it again byte for byte. Needs the corpus oracle set's `skate/shaders` or
//! `elden/shaders` (each a folder, searched recursively, or an archive, of containers as the game ships them).
mod inputs;

mod common;

use inputs::Game;

fn digests(game: Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    for file in &corpus.names {
        let bytes = corpus.read(file);
        let path = corpus.at(file);
        let stored: [u8; 16] = bytes[4..20].try_into().unwrap();
        assert_ne!(stored, [0; 16], "{}: an unsigned container", path);
        assert_eq!(
            dxcontainer::checksum(&bytes),
            Some(stored),
            "{}: the digest differs from the stored one",
            path
        );
        let parts = dxcontainer::chunks(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path));
        // the writer gives the container back byte for byte: the parts in their order, unpadded as stored, the digest
        // it computes equal to the stored one; and a zeroed digest is repaired to the stored one
        assert_eq!(
            dxcontainer::serialize(&parts).unwrap(),
            bytes,
            "{}: re-serialized differently",
            path
        );
        let mut unsigned = bytes.clone();
        unsigned[4..20].fill(0);
        dxcontainer::repair_digest(&mut unsigned).unwrap();
        assert_eq!(unsigned, bytes, "{}: repaired differently", path);
    }
    println!(
        "{}: {} containers, every digest equal to the stored one, every container re-serialized byte for byte",
        corpus.name,
        corpus.names.len()
    );
}

#[test]
fn skate_containers() {
    digests(Game::Skate);
}
#[test]
fn elden_ring_containers() {
    digests(Game::Elden);
}
