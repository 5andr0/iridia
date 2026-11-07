//! A share of a case file's cases, so that CI can run one file on several machines and [`crate::table::make`] put its
//! shards together.

use serde::{Deserialize, Serialize};

/// The cases of a file whose number, divided by `count`, leaves `index`: shard `index` of `count`. The cases of a file
/// are in file order, and one file's runs (its models and runners, one after the other) fall to different shards, so
/// the shards of a file come to about the same work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    /// Which share, from 0.
    pub index: usize,
    /// How many shares there are.
    pub count: usize,
}

impl Shard {
    /// Every case: one shard of one.
    pub const WHOLE: Shard = Shard { index: 0, count: 1 };

    /// The shard `text` names: `index/count`, `index` below `count`.
    pub fn parse(text: &str) -> Result<Shard, String> {
        let usage = || format!("a shard is index/count, as 0/2, with the index below the count: {text}");
        let (index, count) = text.split_once('/').ok_or_else(usage)?;
        let (index, count) = (
            index.parse::<usize>().map_err(|_| usage())?,
            count.parse::<usize>().map_err(|_| usage())?,
        );
        if index >= count {
            return Err(usage());
        }
        Ok(Shard { index, count })
    }

    /// Whether the shard runs case number `case`.
    pub fn has(self, case: usize) -> bool {
        case.checked_rem(self.count)
            .is_some_and(|rest| rest == self.index)
    }

    /// The numbers of the cases the shard runs, of `cases` cases.
    pub fn cases(self, cases: usize) -> Vec<usize> {
        (0..cases).filter(|&i| self.has(i)).collect()
    }
}

impl Default for Shard {
    fn default() -> Self {
        Shard::WHOLE
    }
}

/// The results of the shards of one part, in the order of the cases they ran: `shards[s]` holds shard `s` of
/// `shards.len()`, whose result `j` is case `s + j * shards.len()`. Fails where a shard is not the length its share of
/// the cases gives.
pub fn interleave<T>(shards: Vec<Vec<T>>) -> Result<Vec<T>, String> {
    let count = shards.len();
    let total: usize = shards.iter().map(Vec::len).sum();
    for (s, shard) in shards.iter().enumerate() {
        let share = Shard { index: s, count }.cases(total).len();
        if shard.len() != share {
            return Err(format!(
                "shard {s} of {count} has {} results where {total} cases give it {share}",
                shard.len()
            ));
        }
    }
    let mut iters: Vec<_> = shards.into_iter().map(Vec::into_iter).collect();
    let mut out = Vec::with_capacity(total);
    'rounds: loop {
        for it in &mut iters {
            match it.next() {
                Some(r) => out.push(r),
                None => break 'rounds,
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shard_is_index_over_count() {
        assert_eq!(Shard::parse("1/3"), Ok(Shard { index: 1, count: 3 }));
        for bad in ["", "1", "3/3", "a/2", "1/b", "-1/2", "0/0"] {
            assert!(Shard::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_shards_of_any_number_of_cases_partition_them_and_interleave_back() {
        for count in 1..=5usize {
            for total in 0..=13usize {
                let mut seen = vec![0usize; total];
                let mut shards = Vec::new();
                for index in 0..count {
                    let mine = Shard { index, count }.cases(total);
                    for &c in &mine {
                        seen[c] += 1;
                    }
                    shards.push(mine);
                }
                assert!(seen.iter().all(|&n| n == 1), "{count} shards of {total}");
                let back = interleave(shards).unwrap();
                assert_eq!(back, (0..total).collect::<Vec<_>>(), "{count} shards of {total}");
            }
        }
    }

    #[test]
    fn shards_of_the_wrong_lengths_are_refused() {
        assert!(interleave(vec![vec![0], vec![1, 3]]).is_err());
        assert!(interleave(vec![vec![0, 2, 4], vec![1]]).is_err());
    }
}
