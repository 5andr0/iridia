//! The real-data inputs this crate's tests read, each with what it is, where it comes from and whether a full run
//! needs it (`primase_testdata::inputs!`; README.md here lists them). Integration tests reach it
//! with `mod inputs;`, unit tests with `#[cfg(test)] #[path = "../tests/inputs/mod.rs"] mod inputs;` in the
//! crate root, and tests/real_data_inputs.rs checks that every query the crate's sources write is declared
//! here. Declare an input here when a test first asks for it.

primase_testdata::inputs! {
    games {
        General = "general",
    }
    installs {}
    keys {}
    tools {}
    oracle_sets {
        Corpus = "corpus" {
            about: "shader programs as compilers and games wrote them",
            source: "the user's own installs and public shader corpora, as each program's record states",
        },
    }
    declared {
        Query::Oracle(OracleSet::Corpus, Game::General, "slim") => {
            status: Live,
            gate: true,
            about: r#"the slim shader set: 8,286 DXBC, DXIL and Direct3D 9 programs (`programs/<family>/<sha256>.<family>`, Microsoft's listing beside each, `slim.tsv` saying why each is there) that reach every region and branch side of the lifters the full pool of 108,394 reaches; tests/slim.rs, the pre-push gate"#,
            source: r#"independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4"#,
        },
    }
    debug {}
}
