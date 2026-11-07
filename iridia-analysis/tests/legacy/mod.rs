//! Persistent capped subprocesses for the legacy oracle: an allocation abort must not take the comparison down.
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use super::Answers;

pub(super) const RESOURCE_BUDGET: &str = "legacy oracle refused (resource budget)";

thread_local! {
    static WORKER: RefCell<Option<Worker>> = const { RefCell::new(None) };
}

struct Worker {
    child: Child,
    input: ChildStdin,
    replies: mpsc::Receiver<String>,
    errors: Arc<Mutex<String>>,
    readers: Vec<JoinHandle<()>>,
}

impl Worker {
    fn start() -> Self {
        let python = if cfg!(windows) { "python" } else { "python3" };
        let mut child = Command::new(python)
            .arg("-I")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/legacy/capped.py"))
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "legacy_worker", "--ignored", "--nocapture"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the capped legacy oracle (requires Python 3)");
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let error = child.stderr.take().unwrap();
        let (sender, replies) = mpsc::channel();
        let errors = Arc::new(Mutex::new(String::new()));
        let errors_copy = errors.clone();
        let readers = vec![
            std::thread::spawn(move || {
                for line in BufReader::new(output).lines() {
                    let Ok(line) = line else { break };
                    if let Some(reply) = line.strip_prefix("REPLY\t")
                        && sender.send(reply.to_owned()).is_err()
                    {
                        break;
                    }
                }
            }),
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = error.take(64 * 1024).read_to_end(&mut bytes);
                *errors_copy.lock().unwrap() = String::from_utf8_lossy(&bytes).into_owned();
            }),
        ];
        Self {
            child,
            input,
            replies,
            errors,
            readers,
        }
    }

    fn request(&mut self, bytes: &[u8]) -> Result<Result<Answers, String>, String> {
        let length = u32::try_from(bytes.len()).expect("shader size fits u32");
        self.input
            .write_all(&length.to_le_bytes())
            .and_then(|()| self.input.write_all(bytes))
            .and_then(|()| self.input.flush())
            .map_err(|e| e.to_string())?;
        let reply = self
            .replies
            .recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())?;
        serde_json::from_str(&reply).map_err(|e| format!("legacy protocol: {e}"))
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        for reader in self.readers.drain(..) {
            reader.join().unwrap();
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

pub(super) fn answers(bytes: &[u8]) -> Result<Answers, String> {
    WORKER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let worker = slot.get_or_insert_with(Worker::start);
        match worker.request(bytes) {
            Ok(answer) => answer,
            Err(why) => {
                worker.stop();
                let error = worker.errors.lock().unwrap().clone();
                *slot = None;
                if error.contains("memory allocation") || error.contains("out of memory") {
                    eprintln!("legacy oracle resource budget: memory cap won (512 MiB)\n{error}");
                    Err(RESOURCE_BUDGET.into())
                } else if why.contains("timed out") {
                    eprintln!("legacy oracle resource budget: time cap won (30 s)\n{error}");
                    Err(RESOURCE_BUDGET.into())
                } else {
                    panic!("legacy oracle subprocess failed: {why}\n{error}");
                }
            }
        }
    })
}

/// BTreeMaps with tuple keys travel as sorted entries; JSON objects only accept string keys.
pub(super) mod entries {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<K: Serialize, V: Serialize, S: Serializer>(
        map: &BTreeMap<K, V>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        map.iter().collect::<Vec<_>>().serialize(s)
    }

    pub fn deserialize<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<K, V>, D::Error> {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}
