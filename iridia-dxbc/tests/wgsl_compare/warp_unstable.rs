//! The compute programs WARP gives different words for on different runs of the same job, and which words.
//!
//! Whether a repeat shows the difference is chance: two repeats showed it for one program in two of six identical runs. A class
//! decided by what the repeats show would put the same program in a different class from one run to the next. So the class is
//! decided by this table and the source: a listed program has its listed words left out of the comparison on every run,
//! whatever this run's repeats show, and a program that shows a difference the table does not list fails the test with the
//! row to add. A word the executor's own stores gave two or more different values is not a row's business: WARP's value of it in
//! every repeat must be one of the values the stores gave (`Memory.distinct`), which the test checks on every run for every
//! such word that differs, so those words need neither a row nor a survey. A row is made from surveys at many repeats (`GAME_DATA_DEBUG="shader.compute=only;shader.compute_runs=60;
//! shader.only=<programs>"` on a test binary with the table empty, which prints the rows), and its evidence says how many
//! repeats and which tree it was surveyed on. A row lists every word any survey showed: two surveys on one tree at different
//! machine loads showed different programs (2414 differed in 15 of 60 repeats in one and in none of 236 in the other; load as
//! the cause was not tested), so one survey does not bound the set. The WGSL the writer emits decides the instability: a writer change is a reason to survey again.
use std::collections::{BTreeMap, BTreeSet};

/// Why WARP gives a program's words differently between repeats: a cause the test checks on every run, as a
/// `COMPUTE_EXPECTED` row's shape is checked. A row with no cause shown is not added: the executor and the writer's output
/// must be understood first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cause {
    /// A thread reads a group-shared word another thread writes in the same barrier phase (the lifted IR still has that
    /// shape, `row_shape`): which value it reads is the thread scheduler's.
    SharedScanRace,
    /// Every thread group does the same work on buffer words the program both loads and stores (the lifted IR still has that
    /// shape, `row_shape`): a group's stores land among another's loads, which the executor running them one after the other
    /// does not reproduce. The row's evidence says the words differ between repeats at the dispatched groups and not at one.
    CrossGroupRace,
}

/// A row of [`TABLE`]: name, source hash, words, cause, evidence.
pub(super) type Row = (&'static str, &'static str, &'static str, Cause, &'static str);

/// Programs by name and source hash, the words of each that WARP gives differently between repeats (see [`Words`]), the cause,
/// and the evidence: how many of the repeats of a survey differed from its first, and how many words.
pub(super) const TABLE: &[Row] = &[
    (
        "1806.dxbc",
        "sha256:0f0b6565d9ec923ad2c7407659f94b70e51a70ca38a7d4d57934210c66cd8bc4",
        "14:0-9 14:12-26 14:28-37 14:40-54 14:56-70 14:72-119",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 44 of 60 repeats after the first differed from it over 1 survey logs, 113 words; surveyed on tree res-arrays-700 rebased on fd7fd5a, workgroup zeroing in wgsl-run, stores-overlap words compared as members",
    ),
    (
        "2393.dxbc",
        "sha256:23a0d227f30cdb350d0faed38faab35ead30f4f5bdd94f4acf31eda48fc928a0",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 67 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2395.dxbc",
        "sha256:809960eb776244b98cc625764ac6f5b4e7da892c16b30e5550629e25eaa5773d",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 52 of 1092 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2397.dxbc",
        "sha256:112300fb02430a1b0d98fe3d0565274b2c1801d1c0d7c30d733315badde6f6d8",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 4 of 236 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2400.dxbc",
        "sha256:2eb354422bfb87207a69c8d3c3b76f99a6a855459b24cf0f5c27bb3b1796c9e2",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 16 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2401.dxbc",
        "sha256:995e805c596c3ca4ed010efddd1f4083c7f7d14351a88b9d5fa5d304dd1d51aa",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 46 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2403.dxbc",
        "sha256:71843542541a737abdd199cff0a1ff0f1d0120bfa6ca9a201c681aa30642e5b4",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 34 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2404.dxbc",
        "sha256:d24a8e7c76fabc56f38709e71028bcd0a472f107e475ecf7516e7fe8ee4562bf",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 76 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2405.dxbc",
        "sha256:ba1553e7965eda7484db101d360ac07c865ace232cec46c0c3202d99e4d53729",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 14 of 856 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2406.dxbc",
        "sha256:05b73572d4c27f20c4ced579262e1c05c7eca4d4ef6dc48adb9784730721d348",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 75 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2409.dxbc",
        "sha256:7abbf0c2fbb5bc35bdb5d71f7a48187ca7db3ea794780aabf26a6966501c7016",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 85 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "2414.dxbc",
        "sha256:db65b06a2e8c260fbab0d3e4e1cf35b04eb856bf0a1b6f910dafd28aa40b8a56",
        "1:0-3",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase); 20 of 856 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 4 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "BitonicSortSinglePass.cpo",
        "sha256:bf7ab5b695ec98b090f4f6c9681bfdf3ffdac949424ebc76b9d8b1f0055943ce",
        "0:0-3",
        Cause::CrossGroupRace,
        "cross-group shape (no group number read, dispatched over 2 groups, buffer 0 loaded and stored); the job of the vector that shows it, 1000 replies on one wgsl-run process (sha256 9cce0dcd35a40ef8b3354f800a757fc14ad503ba17a915915cad54d905125ab8): at the dispatch of 2 groups words 0, 1, 2 and 3 of the buffer were off the value most replies gave in 1, 26, 25 and 23 replies, each time another of the four values that the mode has in those words; at 1 group no word differed in 1000 replies; the 16-repeat survey of the four vectors showed 1 of 60 repeats after the first differing in 3 words; surveyed on tree res-arrays-700 rebased on fd7fd5a",
    ),
    (
        "BitonicSortSinglePass128.cpo",
        "sha256:0f573ca50ac42575d0982b5675eca49fd79495e0abacd65cde8b0e9fee7b6eda",
        "0:1 0:3 0:8-11 0:13 0:15-19 0:25 0:27 0:32 0:34 0:40-43 0:48-51 0:56-59 0:64-67 0:72-75 0:80-83 0:88-91 0:96 0:98 0:104-107 0:112-115 0:121 0:123 0:128-131 0:136-139 0:144-147 0:152-155 0:160-163 0:168 0:170 0:176-179 0:184-187 0:193 0:195 0:200-203 0:208-211 0:216 0:218 0:224-227 0:229 0:232-235 0:240-243 0:248-251",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase), cross-group shape (no group number read, dispatched over more than one group); the executor stored different values from two or more invocations to 110 of 110 such words; 135 of 1188 repeats after the first differed from it over 3 survey logs, 115 words; surveyed on tree res-arrays-700 with the workgroup zeroing in wgsl-run (writer unchanged since d55bdce)",
    ),
    (
        "BitonicSortSinglePass256.cpo",
        "sha256:46fc3f9d3a847723d07d736f9714561f69efe3eb852f6556dca5e3ba1ed3de16",
        "0:8-11 0:16-19 0:25 0:27 0:32 0:34 0:40-43 0:48-51 0:56-59 0:64-67 0:72-75 0:80-83 0:88-91 0:96 0:98 0:104-107 0:112-115 0:121 0:123 0:128-131 0:136-139 0:144-147 0:152-155 0:160-163 0:168 0:170 0:176-179 0:184-187 0:193 0:195 0:200-203 0:208-211 0:216 0:218 0:224-227 0:232-235 0:240-243 0:248-251 0:256-259 0:264-267 0:272-275 0:280-283 0:288-291 0:296-299 0:304-307 0:312-315 0:320-323 0:328-331 0:336-339 0:344-347 0:352-355 0:360-363 0:368-371 0:377 0:379 0:384-387 0:392-395 0:400-403 0:408-411 0:416-419 0:425 0:427 0:432-435 0:440-443 0:448-451 0:456 0:458 0:464-467 0:472-475 0:480 0:482 0:488-491 0:496-499 0:504-507",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase), cross-group shape (no group number read, dispatched over more than one group); 51 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 230 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
    (
        "BitonicSortSinglePass512.cpo",
        "sha256:752ec5b2a57ebc63792422a7f84c7fd1f5890161d4c742f2d5bb278469510157",
        "0:8-11 0:16-19 0:25 0:27 0:32 0:34 0:40-43 0:48-51 0:56-59 0:64-67 0:72-75 0:80-83 0:88-91 0:96 0:98 0:104-107 0:112-115 0:121 0:123 0:128-131 0:136-139 0:144-147 0:152-155 0:160-163 0:168 0:170 0:176-179 0:184-187 0:193 0:195 0:200-203 0:208-211 0:216 0:218 0:224-227 0:232-235 0:240-243 0:248-251 0:256-259 0:264-267 0:272-275 0:280-283 0:288-291 0:296-299 0:304-307 0:312-315 0:320-323 0:328-331 0:336-339 0:344-347 0:352-355 0:360-363 0:368-371 0:377 0:379 0:384-387 0:392-395 0:400-403 0:408-411 0:416-419 0:425 0:427 0:432-435 0:440-443 0:448-451 0:456 0:458 0:464-467 0:472-475 0:480 0:482 0:488-491 0:496-499 0:504-507 0:512-515 0:520 0:522 0:528-531 0:536-539 0:544-547 0:552-555 0:560-563 0:568 0:570 0:576-579 0:585 0:587 0:592-595 0:600 0:602 0:608-611 0:616-619 0:624-627 0:632-635 0:640-643 0:648-651 0:656-659 0:664-667 0:672-675 0:680-683 0:689 0:691 0:696-699 0:705 0:707 0:712-715 0:720-723 0:728-731 0:736-739 0:744-747 0:752-755 0:760-763 0:768-771 0:776 0:778 0:785 0:787 0:792-795 0:800-803 0:808 0:810 0:816-819 0:824-827 0:832-835 0:840-843 0:848 0:850 0:856-859 0:865 0:867 0:872-875 0:880 0:882 0:888 0:890 0:896-899 0:904-907 0:912-915 0:920-923 0:928 0:930 0:936-939 0:944-947 0:953 0:955 0:960-963 0:968-971 0:976 0:978 0:984-987 0:992 0:994 0:1000-1003 0:1008-1011 0:1016 0:1018",
        Cause::SharedScanRace,
        "SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase), cross-group shape (no group number read, dispatched over more than one group); 39 of 296 repeats after the first differed from it over surveys of 16, 60 and 200 repeats (4 vectors each; 200 for 9 programs only), 450 words; surveyed on tree 700b3d0 (writer unchanged since d55bdce)",
    ),
];

/// The words of a row: per buffer or image (as the harness labels it: its id, and `[element]` for an element of an array), the
/// ranges of word indices, as `label:first-last` or `label:index`, separated by spaces.
pub(super) struct Words(Vec<(String, u32, u32)>);

impl Words {
    pub fn parse(words: &str) -> Words {
        Words(
            words
                .split_whitespace()
                .map(|w| {
                    let (label, span) = w
                        .rsplit_once(':')
                        .unwrap_or_else(|| panic!("WARP_UNSTABLE words {w:?}: no label"));
                    let (first, last) = span.split_once('-').unwrap_or((span, span));
                    let number = |s: &str| {
                        s.parse::<u32>()
                            .unwrap_or_else(|_| panic!("WARP_UNSTABLE words {w:?}: not a number"))
                    };
                    (label.to_owned(), number(first), number(last))
                })
                .collect(),
        )
    }

    pub fn contains(&self, label: &str, word: u32) -> bool {
        self.0
            .iter()
            .any(|(l, first, last)| l == label && (*first..=*last).contains(&word))
    }

    /// How many words it names.
    pub fn len(&self) -> usize {
        self.0
            .iter()
            .map(|(_, first, last)| (last - first + 1) as usize)
            .sum()
    }
}

/// What the table says of a program of source hash `hash`.
pub(super) enum Listing {
    /// Not listed under this name.
    Unlisted,
    /// Listed under this name for other source: the row is out of date.
    OtherSource,
    /// Listed: its words, its cause and its evidence.
    Listed(Words, Cause, &'static str),
}

pub(super) fn listing(name: &str, hash: &str) -> Listing {
    listing_in(TABLE, name, hash)
}

/// What `table` (rows as [`TABLE`]'s) says of a program: a client with a table of its own reads it with the same rules.
pub(super) fn listing_in(table: &[Row], name: &str, hash: &str) -> Listing {
    let mut other = false;
    for (n, h, words, cause, evidence) in table {
        if *n == super::short(name) {
            if *h == hash {
                return Listing::Listed(Words::parse(words), *cause, evidence);
            }
            other = true;
        }
    }
    if other {
        Listing::OtherSource
    } else {
        Listing::Unlisted
    }
}

/// The words of a row for the words `seen` (by label) that differed between repeats.
pub(super) fn words_of(seen: &BTreeMap<String, BTreeSet<u32>>) -> String {
    let mut out = Vec::new();
    for (label, words) in seen {
        let mut it = words.iter().copied().peekable();
        while let Some(first) = it.next() {
            let mut last = first;
            while it.peek() == Some(&(last + 1)) {
                last = it.next().unwrap_or(last);
            }
            out.push(if first == last {
                format!("{label}:{first}")
            } else {
                format!("{label}:{first}-{last}")
            });
        }
    }
    out.join(" ")
}

/// The row to add to [`TABLE`] for a program that shows words (`seen`) differing between its repeats, `differing` of the
/// `repeats` repeats after the first having differed from it. `cause` is the cause this run shows (`None`: none shown, and the
/// row is printed with a cause that does not exist, so it cannot be pasted as it is); `note` is what else the run found.
pub(super) fn row(
    name: &str,
    hash: &str,
    seen: &BTreeMap<String, BTreeSet<u32>>,
    differing: usize,
    repeats: usize,
    cause: Option<Cause>,
    note: &str,
) -> String {
    let words: usize = seen.values().map(BTreeSet::len).sum();
    let cause = cause.map_or_else(|| "Cause::NONE_SHOWN".to_owned(), |c| format!("Cause::{c:?}"));
    format!(
        "(\"{}\", \"{hash}\", \"{}\", {cause}, \"{note}; {differing} of {repeats} repeats after the first differ from it, {words} words\"),",
        super::short(name),
        words_of(seen)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_round_trip_through_a_row() {
        let seen = BTreeMap::from([
            ("3".to_owned(), BTreeSet::from([0, 1, 2, 7])),
            ("5[1]".to_owned(), BTreeSet::from([40])),
        ]);
        let text = words_of(&seen);
        assert_eq!(text, "3:0-2 3:7 5[1]:40");
        let words = Words::parse(&text);
        assert_eq!(words.len(), 5);
        assert!(words.contains("3", 2) && words.contains("3", 7) && words.contains("5[1]", 40));
        assert!(!words.contains("3", 3) && !words.contains("5", 40) && !words.contains("5[1]", 41));
    }

    #[test]
    fn the_table_parses() {
        for (name, hash, words, _, evidence) in TABLE {
            assert!(hash.starts_with("sha256:"), "{name}");
            assert!(Words::parse(words).len() > 0, "{name}");
            assert!(!evidence.is_empty(), "{name}");
        }
    }
}
