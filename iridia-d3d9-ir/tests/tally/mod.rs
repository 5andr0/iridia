//! Outcome classes counted over a corpus, for the tests that run every program of one and fail on any outcome
//! outside the passing classes.
use std::collections::BTreeMap;

/// The outcome classes of a run over a corpus, each with how many members and the first.
#[derive(Default)]
pub struct Tally(BTreeMap<String, (usize, String)>);

impl Tally {
    pub fn note(&mut self, class: impl Into<String>, example: impl Into<String>) {
        let e = self.0.entry(class.into()).or_insert((0, example.into()));
        e.0 += 1;
    }
    pub fn print(&self, what: &str) {
        println!("{what}:");
        for (class, (n, first)) in &self.0 {
            println!("{n:7} {class}\n          e.g. {first}");
        }
    }
    /// How many are in class `class`.
    #[allow(dead_code, reason = "not every test that counts pins a class")]
    pub fn count(&self, class: &str) -> usize {
        self.0.get(class).map_or(0, |(n, _)| *n)
    }
    /// The members of the classes `fails` picks.
    #[allow(dead_code, reason = "not every test picks its failures by name")]
    pub fn failing(&self, fails: impl Fn(&str) -> bool) -> usize {
        self.0
            .iter()
            .filter(|(class, _)| fails(class))
            .map(|(_, (n, _))| n)
            .sum()
    }
    /// The members outside the classes `passing`.
    #[allow(dead_code, reason = "not every test names its passing classes")]
    pub fn failures(&self, passing: &[&str]) -> usize {
        self.0
            .iter()
            .filter(|(class, _)| !passing.contains(&class.as_str()))
            .map(|(_, (n, _))| n)
            .sum()
    }
}

/// A class key with its numbers taken out, so one failure kind is one class.
pub fn class(e: &str) -> String {
    e.chars()
        .map(|c| if c.is_ascii_digit() { 'N' } else { c })
        .collect()
}
