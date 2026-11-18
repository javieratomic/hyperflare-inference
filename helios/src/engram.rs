/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod cache;
#[cfg(unix)]
mod prepared;
mod reader;
use crate::{
    Error, Result,
    cuda::{Event, PinnedHost, Stream},
    execution::{ExecutionManifest, GraphKind, PromptWindow, Upload},
};
#[cfg(unix)]
pub use prepared::{PlaneSource, PreparedTables, SourceSegment, TableSource};
pub(crate) use reader::Plane;
#[cfg(feature = "server")]
use std::fmt::Write;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Default)]
pub(crate) struct Stats {
    gathers: AtomicU64,
    gather_ns: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    resident_rows: AtomicU64,
    capacity_rows: AtomicU64,
    cache_bytes: AtomicU64,
    resident_scale_bytes: AtomicU64,
    disk_reads: AtomicU64,
    disk_bytes: AtomicU64,
    disk_plan_ns: AtomicU64,
    disk_submit_ns: AtomicU64,
    disk_wait_ns: AtomicU64,
    disk_complete_ns: AtomicU64,
    prefetch_step_rows: AtomicU64,
    prefetch_capacity_rows: AtomicU64,
    prefetch_rows: AtomicU64,
    prefetch_hits: AtomicU64,
    prefetch_unused: AtomicU64,
    prefetch_interrupted: AtomicU64,
    prefetch_errors: AtomicU64,
    prefetch_ns: AtomicU64,
}

impl Stats {
    #[cfg(feature = "server")]
    pub(crate) fn render(&self, text: &mut String) {
        for (name, kind, help, counter) in [
            (
                "gathers_total",
                "counter",
                "Embedding row gathers, excluding the wait for GPU-produced IDs.",
                &self.gathers,
            ),
            (
                "gather_nanoseconds_total",
                "counter",
                "Wall time inside embedding row gathers.",
                &self.gather_ns,
            ),
            (
                "row_cache_hits_total",
                "counter",
                "Cached row hits; duplicate IDs within one gather count once.",
                &self.hits,
            ),
            (
                "row_cache_misses_total",
                "counter",
                "Cached row misses; duplicate IDs within one gather count once.",
                &self.misses,
            ),
            (
                "row_cache_evictions_total",
                "counter",
                "Resident rows replaced by other row IDs.",
                &self.evictions,
            ),
            (
                "row_cache_resident_rows",
                "gauge",
                "Rows held in the embedding row cache.",
                &self.resident_rows,
            ),
            (
                "row_cache_capacity_rows",
                "gauge",
                "Maximum rows held in the embedding row cache.",
                &self.capacity_rows,
            ),
            (
                "row_cache_bytes",
                "gauge",
                "Allocated embedding row payload, usage counters, index, bounded miss workspace and read-ahead row bytes.",
                &self.cache_bytes,
            ),
            (
                "resident_scale_bytes",
                "gauge",
                "Resident scale-plane allocation bytes for disk-backed embedding tables, separate from the row cache.",
                &self.resident_scale_bytes,
            ),
            (
                "disk_reads_total",
                "counter",
                "Direct-I/O embedding read requests queued.",
                &self.disk_reads,
            ),
            (
                "disk_bytes_total",
                "counter",
                "Bytes requested by direct-I/O embedding reads.",
                &self.disk_bytes,
            ),
            (
                "disk_plan_nanoseconds_total",
                "counter",
                "Wall time building, sorting and coalescing disk row reads.",
                &self.disk_plan_ns,
            ),
            (
                "disk_submit_nanoseconds_total",
                "counter",
                "Cumulative worker wall time preparing and issuing disk read batches; Linux kernel submission is included in wait time.",
                &self.disk_submit_ns,
            ),
            (
                "disk_wait_nanoseconds_total",
                "counter",
                "Cumulative worker wall time in disk completion waits, including Linux io_uring submission.",
                &self.disk_wait_ns,
            ),
            (
                "disk_complete_nanoseconds_total",
                "counter",
                "Cumulative worker wall time processing disk completions and copying rows, excluding completion waits.",
                &self.disk_complete_ns,
            ),
            (
                "prefetch_step_rows",
                "gauge",
                "Rows that one step of the widest graph requests from the disk tables whose rows the host can derive; zero when there is none.",
                &self.prefetch_step_rows,
            ),
            (
                "prefetch_capacity_rows",
                "gauge",
                "Rows that read-ahead can hold within the row-cache budget, at most prefetch_step_rows; zero when no table reads ahead.",
                &self.prefetch_capacity_rows,
            ),
            (
                "prefetch_rows_total",
                "counter",
                "Rows read from disk ahead of the gathers predicted to request them, excluding rows already cached or held from an earlier prediction.",
                &self.prefetch_rows,
            ),
            (
                "prefetch_unused_total",
                "counter",
                "Rows read ahead and dropped before any gather took them: an announcement stopped listing them, the rows wanted soonest needed their place, or the table stopped reading ahead. A derivation that differs from the graph's shows here: these rows approach prefetch_rows_total while prefetch_hits_total stays near zero.",
                &self.prefetch_unused,
            ),
            (
                "prefetch_hits_total",
                "counter",
                "Cached row misses served from rows read ahead instead of disk reads inside the gather; duplicate IDs within one gather count once.",
                &self.prefetch_hits,
            ),
            (
                "prefetch_interrupted_total",
                "counter",
                "Read-ahead passes cut short because a gather was submitted or the table's delivery was cancelled; the rows already read stay available.",
                &self.prefetch_interrupted,
            ),
            (
                "prefetch_errors_total",
                "counter",
                "Read-ahead passes ended by a failed read, whose rows are not held and are read again by the gather that needs them, or by a panic, after which the table stops reading ahead.",
                &self.prefetch_errors,
            ),
            (
                "prefetch_nanoseconds_total",
                "counter",
                "Wall time reading rows ahead, outside embedding row gathers; its disk reads also count in the disk counters.",
                &self.prefetch_ns,
            ),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_engram_{name} {help}\n# TYPE flare_engram_{name} {kind}\nflare_engram_{name} {}",
                counter.load(Ordering::Relaxed)
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Storage {
    #[default]
    Disk,
    Ram,
}

/// One lookup head of an n-gram table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Head {
    /// Tokens in the n-gram: the current token and `order - 1` preceding ones.
    pub(crate) order: usize,
    pub(crate) prime: u64,
    /// First row of the head in this rank's planes; `None` when another rank stores it.
    pub(crate) first_row: Option<u64>,
}

/// XOR-modulo row derivation of one table, as its captured graphs compute it. A head of
/// order `n` XORs the products of the `n` most recent normalized token IDs with their lag
/// multipliers, reduces the result by the head's prime, and adds the head's first row.
/// From the first token outside the normalized vocabulary backward, every token of the
/// n-gram counts as padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowHash {
    /// Normalized vocabulary ID of each raw token ID.
    pub(crate) token_map: Arc<[u32]>,
    /// Raw token ID standing in for prompt positions before the first token.
    pub(crate) pad: u32,
    /// Multipliers by lag: the current token first, then each preceding token.
    pub(crate) multipliers: Vec<u64>,
    /// Heads in output-column order.
    pub(crate) heads: Vec<Head>,
}

impl RowHash {
    /// Preceding prompt tokens that enter the longest n-gram.
    pub(crate) fn history(&self) -> usize {
        self.multipliers.len().saturating_sub(1)
    }

    /// Heads whose rows this rank stores: the rows one token requests from it.
    fn local_heads(&self) -> usize {
        self.heads
            .iter()
            .filter(|head| head.first_row.is_some())
            .count()
    }

    /// Append the rows of [`Self::each_row`] until `rows` holds `limit` IDs.
    #[cfg(test)]
    pub(crate) fn rows(&self, history: &[u32], tokens: &[u32], limit: usize, rows: &mut Vec<i64>) {
        self.each_row(history, tokens, &mut |row: i64| {
            if rows.len() >= limit {
                return false;
            }
            rows.push(row);
            true
        });
    }

    /// Report, in token order, the rows this rank stores for `tokens`, which follow
    /// `history` in one prompt. Positions before the prompt read as `pad`. Stops and
    /// returns false once `emit` does.
    fn each_row(
        &self,
        history: &[u32],
        tokens: &[u32],
        emit: &mut dyn FnMut(i64) -> bool,
    ) -> bool {
        let depth = self.history();
        let history = &history[history.len().saturating_sub(depth)..];
        let missing = depth - history.len();
        let normalized =
            |token: u32| self.token_map.get(token as usize).map(|id| u64::from(*id));
        let Some(pad) = normalized(self.pad) else {
            return true;
        };
        // Lag `lag` of the token at `position` reads `source(position + depth - lag)`.
        let source = |index: usize| {
            if index < missing {
                Some(pad)
            } else if index < depth {
                normalized(history[index - missing])
            } else {
                normalized(tokens[index - depth])
            }
        };
        let mut mixed = vec![0u64; self.multipliers.len()];
        for position in 0..tokens.len() {
            let mut rolling = 0u64;
            let mut blocked = false;
            for (lag, multiplier) in self.multipliers.iter().enumerate() {
                let token = source(position + depth - lag).filter(|_| !blocked);
                blocked = token.is_none();
                rolling ^= token.unwrap_or(pad).wrapping_mul(*multiplier);
                mixed[lag] = rolling;
            }
            for head in &self.heads {
                let Some(first_row) = head.first_row else {
                    continue;
                };
                if !emit((first_row + mixed[head.order - 1] % head.prime) as i64) {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Table {
    pub(crate) planes: [Plane; 2],
    pub(crate) heads: usize,
    pub(crate) weights: String,
    pub(crate) scales: String,
    pub(crate) hashes: String,
    pub(crate) event: String,
    pub(crate) request: Option<String>,
    /// Present when the host can derive the table's row IDs from prompt tokens.
    pub(crate) hash: Option<RowHash>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Index {
    pub(crate) tables: Vec<Table>,
}

impl Table {
    fn staging_payload_bytes(&self, max_tokens: usize) -> Result<usize> {
        self.planes.iter().try_fold(0usize, |total, plane| total.checked_add(plane.row_bytes))
            .and_then(|bytes| bytes.checked_mul(self.heads))
            .and_then(|bytes| bytes.checked_mul(max_tokens))
            .ok_or_else(|| "embedding staging exceeds host range".into())
    }
}

pub(crate) fn staging_bytes(index: &Index, max_tokens: usize) -> Result<u64> {
    index.tables.iter().try_fold(0u64, |total, table| {
        let payload = table.staging_payload_bytes(max_tokens)? as u64;
        payload.checked_add(4).and_then(|bytes| bytes.checked_mul(STAGING_FRAMES as u64))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| "embedding staging exceeds the memory budget".into())
    })
}

struct Staging {
    data: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}
impl Staging {
    fn signal(&self, generation: u32) {
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}

/// Where a table reader delivers gathered rows. Serving delivers into the pinned staging
/// frames that the GPU copies from; tests deliver into host memory.
trait Frames {
    /// The first `nbytes` row bytes of `frame`.
    ///
    /// # Safety
    /// The frame holds at least `nbytes`, and the caller is its only user from this call
    /// until it signals the frame.
    #[allow(clippy::mut_from_ref)]
    unsafe fn rows(&self, frame: usize, nbytes: usize) -> &mut [u8];
    /// Publish that `frame` holds the rows of `generation`, or that their delivery failed.
    fn signal(&self, frame: usize, generation: u32);
    /// Record a failure of the table's delivery for submissions to report.
    fn fail(&self, frame: usize, error: Error);
}

impl Frames for [Staging; STAGING_FRAMES] {
    unsafe fn rows(&self, frame: usize, nbytes: usize) -> &mut [u8] {
        // SAFETY: Each frame stays owned until every rank retires its consuming graph, and
        // the reader alone writes its data before publishing the ready flag.
        unsafe { std::slice::from_raw_parts_mut(self[frame].data.address() as *mut u8, nbytes) }
    }
    fn signal(&self, frame: usize, generation: u32) {
        self[frame].signal(generation);
    }
    fn fail(&self, frame: usize, error: Error) {
        *self[frame].error.lock().unwrap() = Some(error);
    }
}
enum Requests {
    Host(Vec<i64>),
    Device {
        address: usize,
        flag: usize,
        sequence: u64,
        rows: usize,
    },
}
impl Requests {
    fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
    fn read(self, cancelled: &AtomicBool) -> Result<Vec<i64>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device {
                address,
                flag,
                sequence,
                rows,
            } => {
                let ready = unsafe { &*(flag as *const AtomicU64) };
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    std::hint::spin_loop();
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
const STAGING_FRAMES: usize = 2;

struct Job {
    frame: usize,
    generation: u32,
    sequence: u64,
    requests: Requests,
}

/// Prompt tokens announced before the graphs that will hash them are submitted.
struct Announcement {
    /// Submission sequence whose gather the read-ahead follows.
    after: u64,
    /// Windows in the order their rows are wanted: for each, the tokens preceding the
    /// announced ones, then the announced tokens.
    windows: Vec<(Vec<u32>, Vec<u32>)>,
}

enum Message {
    Gather(Job),
    /// An announcement was left for the reader, which takes it from its feed.
    Announced,
}

/// What a reader does about the latest announcement once it has served `delivered`.
enum Due {
    /// Nothing to read: no announcement, one that waits for its gather, or one that expired.
    Idle,
    /// The announcement follows the gather just served, but a gather is in flight.
    Wait,
    Read(Arc<Announcement>),
}

/// Decide about the latest announcement once the gather for `delivered` has been served.
/// An announcement waits for the gather it follows, so that its reads overlap that gather's
/// graph instead of delaying the rows the graph waits for. It also waits while any gather
/// of the group is in flight, and expires once a later gather has started: its rows were
/// predicted for the steps after the gather it follows, one of which is then running. A
/// later step replaces it only if that step announces prompt windows itself.
fn due(announced: &mut Option<Arc<Announcement>>, delivered: u64, gathering: bool) -> Due {
    let Some(after) = announced.as_ref().map(|announcement| announcement.after) else {
        return Due::Idle;
    };
    match after.cmp(&delivered) {
        std::cmp::Ordering::Greater => Due::Idle,
        std::cmp::Ordering::Less => {
            *announced = None;
            Due::Idle
        }
        std::cmp::Ordering::Equal if gathering => Due::Wait,
        std::cmp::Ordering::Equal => announced.take().map_or(Due::Idle, Due::Read),
    }
}

/// The announcement that rank `rank` of a delivery group leaves for the group's readers:
/// `windows` in order, ending after `listed` tokens. The ranks of a group share its
/// readers, so only the first rank's call makes one, and windows without tokens make
/// none.
fn announcement(
    rank: usize,
    after: u64,
    windows: &[PromptWindow<'_>],
    listed: usize,
) -> Option<Arc<Announcement>> {
    if rank != 0 {
        return None;
    }
    let mut budget = listed;
    let mut owned = Vec::new();
    for window in windows {
        let count = window.tokens.len().min(budget);
        if count != 0 {
            owned.push((window.history.to_vec(), window.tokens[..count].to_vec()));
            budget -= count;
        }
    }
    (!owned.is_empty()).then(|| {
        Arc::new(Announcement {
            after,
            windows: owned,
        })
    })
}

/// Hold the rows that `hash` derives from the announced prompt windows, reading those
/// not yet held. Each window is one part of the prediction, so the rows of an earlier
/// window are read before those of a later one.
fn read_ahead(
    reader: &mut reader::Reader,
    planes: &[Plane; 2],
    hash: &RowHash,
    announcement: &Announcement,
    interrupted: &dyn Fn() -> bool,
) -> Result<()> {
    reader.read_ahead(
        planes,
        |sink| {
            for (part, (history, tokens)) in announcement.windows.iter().enumerate() {
                if !hash.each_row(history, tokens, &mut |row: i64| sink(part as u32, row)) {
                    break;
                }
            }
        },
        interrupted,
    )
}

/// The submitting side of one table's reader thread.
struct Feed {
    sender: Option<mpsc::SyncSender<Message>>,
    cancelled: Arc<AtomicBool>,
    /// Gathers submitted to the tables of one delivery group and not yet delivered. The
    /// tables share it: read-ahead for one table leaves the disk to a gather of another.
    gathering: Arc<AtomicUsize>,
    /// Most recent announcement the reader has not yet taken.
    announced: Arc<Mutex<Option<Arc<Announcement>>>>,
    /// Whether an `Announced` message is in the channel. At most one is, so it uses the
    /// slot beyond the staging frames and gathers find the channel as free as without it.
    notified: Arc<AtomicBool>,
    /// Whether the reader can derive row IDs from prompt tokens and hold their rows.
    reads_ahead: bool,
    /// A failure outside any gather that left the reader unusable, for the next
    /// submission and the gathers already queued to report. A gather's own failure is
    /// recorded on its frame alone.
    failure: Arc<Mutex<Option<Error>>>,
}

/// The reader thread's side of a [`Feed`].
struct Source {
    receiver: mpsc::Receiver<Message>,
    failure: Arc<Mutex<Option<Error>>>,
    cancelled: Arc<AtomicBool>,
    gathering: Arc<AtomicUsize>,
    announced: Arc<Mutex<Option<Arc<Announcement>>>>,
    notified: Arc<AtomicBool>,
}

/// Connect a reader thread to its submitting side. `gathering` counts the gathers in
/// flight for every table of the delivery group.
fn link(gathering: &Arc<AtomicUsize>, reads_ahead: bool) -> (Feed, Source) {
    let (sender, receiver) = mpsc::sync_channel(STAGING_FRAMES + 1);
    let cancelled = Arc::new(AtomicBool::new(false));
    let announced = Arc::new(Mutex::new(None));
    let notified = Arc::new(AtomicBool::new(false));
    let failure = Arc::new(Mutex::new(None));
    let feed = Feed {
        sender: Some(sender),
        cancelled: cancelled.clone(),
        gathering: gathering.clone(),
        announced: announced.clone(),
        notified: notified.clone(),
        reads_ahead,
        failure: failure.clone(),
    };
    let source = Source {
        receiver,
        failure,
        cancelled,
        gathering: gathering.clone(),
        announced,
        notified,
    };
    (feed, source)
}

impl Feed {
    /// Submit a gather. It is in flight from here until its frame is signalled, so the
    /// read-ahead of every table of the group yields to it at once.
    fn gather(&self, job: Job) -> Result<()> {
        self.gathering.fetch_add(1, Ordering::AcqRel);
        let sent = self
            .sender
            .as_ref()
            .is_some_and(|sender| sender.send(Message::Gather(job)).is_ok());
        if !sent {
            self.gathering.fetch_sub(1, Ordering::AcqRel);
            return Err("Engram reader disconnected".into());
        }
        Ok(())
    }

    /// Leave `announcement` for the reader, replacing one it has not taken, and wake it.
    /// Never blocks.
    fn announce(&self, announcement: &Arc<Announcement>) {
        if !self.reads_ahead || self.cancelled.load(Ordering::Acquire) {
            return;
        }
        *self.announced.lock().unwrap() = Some(announcement.clone());
        if !self.notified.swap(true, Ordering::AcqRel)
            && self
                .sender
                .as_ref()
                .is_none_or(|sender| sender.try_send(Message::Announced).is_err())
        {
            self.notified.store(false, Ordering::Release);
        }
    }
}

/// How long a reader whose announcement is due waits before asking again whether the gathers
/// in flight for the group's other tables have been delivered.
const GATHERING_POLL: Duration = Duration::from_micros(500);

/// Run one table's reader until its feed is dropped: deliver each gather into its frame
/// and, between gathers, read ahead the rows of the latest announcement.
fn serve<F: Frames + ?Sized>(
    source: Source,
    mut reader: reader::Reader,
    planes: [Plane; 2],
    mut hash: Option<RowHash>,
    frames: &F,
) {
    let Source {
        receiver,
        failure,
        cancelled: stop,
        gathering,
        announced,
        notified,
    } = source;
    let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
    // Sequence of the most recent gather; submission sequences start at one.
    let mut delivered = 0;
    // Whether an announcement is due and waits for the gathers in flight.
    let mut waiting = false;
    loop {
        let message = if waiting {
            match receiver.recv_timeout(GATHERING_POLL) {
                Ok(message) => Some(message),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            }
        } else {
            match receiver.recv() {
                Ok(message) => Some(message),
                Err(_) => return,
            }
        };
        match message {
            Some(Message::Gather(job)) => {
                delivered = job.sequence;
                // SAFETY: A frame is submitted again only after every rank retired the
                // graph consuming it, and this thread is the table's only reader.
                let output = unsafe { frames.rows(job.frame, job.requests.rows() * row_bytes) };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if stop.load(Ordering::Acquire) {
                        // A gather that reaches a reader left unusable by read-ahead
                        // reports that failure, not a cancellation.
                        let unusable = failure.lock().unwrap().clone();
                        return Err(unusable.unwrap_or_else(|| "n-gram request cancelled".into()));
                    }
                    let ids = job.requests.read(&stop)?;
                    reader.gather(&planes, &ids, output)
                }))
                .unwrap_or_else(|_| Err("Engram reader panicked".into()));
                if let Err(error) = result {
                    output.fill(0);
                    frames.fail(job.frame, error);
                    stop.store(true, Ordering::Release);
                }
                // Release failed reads too, so graph shutdown can drain before reporting the error.
                frames.signal(job.frame, job.generation);
                gathering.fetch_sub(1, Ordering::AcqRel);
            }
            // Clear before reading the announcement, so one that arrives after this point
            // sends its own notice.
            Some(Message::Announced) => notified.store(false, Ordering::Release),
            None => {}
        }
        waiting = false;
        let Some(derivation) = &hash else { continue };
        if stop.load(Ordering::Acquire) {
            continue;
        }
        let busy = gathering.load(Ordering::Acquire) != 0;
        let announcement = match due(&mut announced.lock().unwrap(), delivered, busy) {
            Due::Idle => continue,
            Due::Wait => {
                waiting = true;
                continue;
            }
            Due::Read(announcement) => announcement,
        };
        let interrupted =
            || gathering.load(Ordering::Acquire) != 0 || stop.load(Ordering::Acquire);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_ahead(&mut reader, &planes, derivation, &announcement, &interrupted)
        }))
        .unwrap_or_else(|_| Err("Engram reader panicked".into()));
        let Err(error) = result else { continue };
        if reader.drop_read_ahead() {
            // The pass ended outside its disk reads, which all completed: a panic in the
            // row derivation or in the bookkeeping of held rows. Gathers use neither, so
            // the table keeps serving them and reads no more rows ahead.
            eprintln!("[flare] embedding row read-ahead stopped for one table: {error}");
            hash = None;
        } else {
            // Reads left outstanding make this reader unusable. No gather is at fault,
            // so the error waits in the table's own slot for the next submission. A
            // step whose rows were already delivered is not failed; a gather already
            // queued fails with this error.
            *failure.lock().unwrap() = Some(error);
            stop.store(true, Ordering::Release);
        }
    }
}

#[derive(Clone, Copy, Default)]
struct State {
    sequence: u64,
    generation: u32,
    rows: usize,
    submitted: usize,
    completed: usize,
}
impl State {
    fn prepare(&mut self, rank: usize, sequence: u64, rows: usize, ranks: usize) -> Result<bool> {
        if sequence == 0 || rank >= ranks {
            return Err("invalid Engram submission identity".into());
        }
        let fresh = self.sequence != sequence;
        if fresh {
            if self.sequence > sequence
                || (self.sequence != 0 && self.completed != (1 << ranks) - 1)
            {
                return Err("Engram staging reused before all ranks completed".into());
            }
            *self = Self {
                sequence,
                generation: self.generation.wrapping_add(1),
                rows,
                submitted: 0,
                completed: 0,
            };
        }
        if self.rows != rows {
            return Err("ranks submitted different Engram row counts".into());
        }
        if self.submitted & (1 << rank) != 0 {
            return Err("rank submitted Engram generation twice".into());
        }
        self.submitted |= 1 << rank;
        Ok(fresh)
    }

    fn finish(&mut self, rank: usize, sequence: u64) -> Result<()> {
        if self.sequence != sequence || self.submitted & (1 << rank) == 0 {
            return Err("Engram completion does not match its submission".into());
        }
        if self.completed & (1 << rank) != 0 {
            return Err("rank completed Engram generation twice".into());
        }
        self.completed |= 1 << rank;
        Ok(())
    }
}

struct Layer {
    staging: Arc<[Staging; STAGING_FRAMES]>,
    feed: Feed,
    worker: Option<JoinHandle<()>>,
    state: Mutex<[State; STAGING_FRAMES]>,
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.feed.cancelled.store(true, Ordering::Release);
        self.feed.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Steps of the widest graph that one announcement may list. It lists more than the
/// read-ahead holds, so that rows held for one prompt stay listed while the steps of
/// other prompts run.
const LISTED_STEPS: usize = 4;

pub(crate) struct Shared {
    pub index: Arc<Index>,
    layers: Vec<Layer>,
    ranks: usize,
    max_tokens: usize,
    stats: Arc<Stats>,
}
impl Shared {
    #[cfg(test)]
    pub(crate) fn layout_fixture(index: Index, ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            index: Arc::new(index),
            layers: Vec::new(),
            ranks,
            max_tokens: 4096,
            stats: Arc::new(Stats::default()),
        })
    }

    pub fn new(
        index: Index,
        files: &[Arc<std::fs::File>],
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        storage: Storage,
        cache_bytes: u64,
        workers: usize,
        resident_scales: bool,
    ) -> Result<Arc<Self>> {
        if storage == Storage::Ram && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        let index = Arc::new(index);
        let stats = Arc::new(Stats::default());
        // Rows that one step of the widest graph requests from a disk table whose rows the
        // host can derive. The Windows readers do not read ahead.
        let step_rows = |table: &Table| {
            table
                .hash
                .as_ref()
                .filter(|_| storage == Storage::Disk && cfg!(unix))
                .map_or(0, |hash| max_tokens.saturating_mul(hash.local_heads()))
        };
        stats.prefetch_step_rows.store(
            index.tables.iter().map(|table| step_rows(table) as u64).sum(),
            Ordering::Relaxed,
        );
        let table_bytes: u128 = index
            .tables
            .iter()
            .flat_map(|table| &table.planes)
            .map(|plane| plane.nbytes as u128)
            .sum();
        let readers = index
            .tables
            .iter()
            .map(|table| {
                let bytes: u128 = table.planes.iter().map(|plane| plane.nbytes as u128).sum();
                let budget = usize::try_from(cache_bytes as u128 * bytes / table_bytes)
                    .map_err(|_| "embedding row cache exceeds host address range")?;
                let rows = max_tokens
                    .checked_mul(table.heads)
                    .ok_or("embedding row capacity exceeds host range")?;
                reader::Reader::from_files(
                    files,
                    &table.planes,
                    storage == Storage::Ram,
                    budget,
                    rows,
                    stats.clone(),
                    workers,
                    resident_scales,
                    // One step of the rows this rank stores; other ranks hold the rest.
                    step_rows(table),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_readers(index, readers, ranks, max_tokens, context, device, stats)
    }

    fn with_readers(
        index: Arc<Index>,
        readers: Vec<reader::Reader>,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        stats: Arc<Stats>,
    ) -> Result<Arc<Self>> {
        if ranks == 0 || ranks > 16 || max_tokens == 0 || readers.len() != index.tables.len() {
            return Err("invalid embedding delivery geometry".into());
        }
        let stream = Stream::open(context, device)?;
        let mut layers = Vec::new();
        let gathering = Arc::new(AtomicUsize::new(0));
        for (table, reader) in index.tables.iter().zip(readers) {
            let staging_bytes = table.staging_payload_bytes(max_tokens)?;
            let mut staging = Vec::with_capacity(STAGING_FRAMES);
            for _ in 0..STAGING_FRAMES {
                staging.push(Staging {
                    data: stream.mapped_host_buffer(staging_bytes)?,
                    ready: stream.mapped_host_buffer(4)?,
                    error: Mutex::new(None),
                });
            }
            let staging: Arc<[Staging; STAGING_FRAMES]> = Arc::new(
                staging
                    .try_into()
                    .map_err(|_| "invalid Engram staging frame count")?,
            );
            let hash = table.hash.clone().filter(|_| reader.reads_ahead());
            let (feed, source) = link(&gathering, hash.is_some());
            let frames = staging.clone();
            let planes = table.planes;
            let worker = thread::Builder::new()
                .name(format!("flare-{}", table.event))
                .spawn(move || serve(source, reader, planes, hash, &*frames))
                .map_err(|e| format!("start Engram reader: {e}"))?;
            layers.push(Layer {
                staging,
                feed,
                worker: Some(worker),
                state: Mutex::new(std::array::from_fn(|_| State::default())),
            });
        }
        Ok(Arc::new(Self {
            index,
            layers,
            ranks,
            max_tokens,
            stats,
        }))
    }

    #[cfg(feature = "server")]
    pub(crate) fn stats(&self) -> Arc<Stats> {
        self.stats.clone()
    }

    pub(crate) fn cache_bytes(&self) -> u64 {
        self.stats.cache_bytes.load(Ordering::Relaxed)
    }

    /// The rows that one step of the widest graph requests from the tables that could
    /// read ahead, and the rows their read-ahead holds within the row-cache budget.
    pub(crate) fn read_ahead_rows(&self) -> (u64, u64) {
        (
            self.stats.prefetch_step_rows.load(Ordering::Relaxed),
            self.stats.prefetch_capacity_rows.load(Ordering::Relaxed),
        )
    }

    /// What an announcement can use, or `None` when the catalog has no table whose rows
    /// are derived from prompt tokens: the preceding prompt tokens that each window must
    /// carry for the longest n-gram, the tokens of one step of the widest graph, whose
    /// rows a read-ahead holds at most, and the most tokens one announcement may list.
    /// The catalog is the same on every rank; [`Self::reads_ahead`] tells whether this
    /// delivery group, with its host's table storage and row cache, uses announcements.
    pub(crate) fn read_ahead_extent(&self) -> Option<(usize, usize, usize)> {
        self.index
            .tables
            .iter()
            .filter_map(|table| table.hash.as_ref())
            .map(RowHash::history)
            .max()
            .map(|history| {
                (
                    history,
                    self.max_tokens,
                    self.max_tokens.saturating_mul(LISTED_STEPS),
                )
            })
    }

    /// Whether a table of this delivery group reads rows ahead.
    pub(crate) fn reads_ahead(&self) -> bool {
        self.layers.iter().any(|layer| layer.feed.reads_ahead)
    }

    /// Announce prompt tokens that the submissions following `after` are expected to
    /// prefill, in the order their rows are wanted. Each table reader able to read ahead
    /// derives their rows and, once it has served the gather for `after` and no gather of
    /// the group is in flight, reads from disk those it does not already hold, while
    /// that gather's graph runs. Rows held for an earlier announcement stay held while
    /// this one lists them. The announcement is a prediction: it never blocks, replaces
    /// an earlier one not yet read, and does not change which rows a graph receives.
    fn announce(&self, rank: usize, after: u64, windows: &[PromptWindow<'_>]) {
        if !self.reads_ahead() {
            return;
        }
        let listed = self.max_tokens.saturating_mul(LISTED_STEPS);
        if let Some(announcement) = announcement(rank, after, windows, listed) {
            for layer in &self.layers {
                layer.feed.announce(&announcement);
            }
        }
    }
    fn validate_start(&self, table: usize, rank: usize, sequence: u64, rows: usize) -> Result<()> {
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let states = layer.state.lock().unwrap();
        if let Some(error) = layer.feed.failure.lock().unwrap().as_ref() {
            return Err(error.clone());
        }
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let mut state = states[sequence as usize % STAGING_FRAMES];
        state.prepare(rank, sequence, rows, self.ranks)?;
        Ok(())
    }

    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let mut states = layer.state.lock().unwrap();
        if let Some(error) = layer.feed.failure.lock().unwrap().as_ref() {
            return Err(error.clone());
        }
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let frame = sequence as usize % STAGING_FRAMES;
        let mut state = states[frame];
        let fresh = state.prepare(rank, sequence, rows, self.ranks)?;
        if fresh {
            layer.feed.gather(Job {
                frame,
                generation: state.generation,
                sequence,
                requests,
            })?;
        }
        states[frame] = state;
        Ok((rows, state.generation))
    }
    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        let mut error = None;
        for layer in &self.layers {
            let mut states = layer.state.lock().unwrap();
            let frame = sequence as usize % STAGING_FRAMES;
            states[frame].finish(rank, sequence)?;
            error = error.or_else(|| layer.staging[frame].error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) events: Vec<usize>,
}
impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramBinding")
            .field("rank", &self.rank)
            .finish()
    }
}
struct Transfer {
    stream: Stream,
    flags: [usize; STAGING_FRAMES],
    weights: usize,
    scales: usize,
    event: usize,
    request: Option<(usize, usize)>,
}
pub(crate) struct Transfers {
    binding: Binding,
    layers: Vec<Transfer>,
}
impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        if let Some(buffer) = manifest
            .buffers
            .iter()
            .find(|buffer| buffer.name == "readiness_timestamps")
        {
            if buffer.shape[1] / 3 != binding.shared.index.tables.len() {
                return Err(
                    "readiness interval count differs from the embedding table catalog".into(),
                );
            }
        }
        let mut layers = Vec::new();
        for (i, table) in binding.shared.index.tables.iter().enumerate() {
            let stream = Stream::open(manifest.context, manifest.device)?;
            let mut flags = [0; STAGING_FRAMES];
            for (flag, staging) in flags
                .iter_mut()
                .zip(binding.shared.layers[i].staging.iter())
            {
                *flag = stream.mapped_address(&staging.ready)?;
            }
            let request = table
                .request
                .as_ref()
                .map(|name| {
                    let buffer = manifest.buffer(name)?;
                    let address = stream.host_address(buffer.address, buffer.nbytes)?;
                    Ok::<_, Error>((address, address + buffer.nbytes - 8))
                })
                .transpose()?;
            layers.push(Transfer {
                stream,
                flags,
                weights: manifest.buffer(&table.weights)?.address,
                scales: manifest.buffer(&table.scales)?.address,
                event: binding.events[i],
                request,
            });
        }
        Ok(Self { binding, layers })
    }
    pub fn validate(&self, sequence: u64, tokens: usize, uploads: &[Upload<'_>]) -> Result<()> {
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let rows = if transfer.request.is_some() {
                tokens
                    .checked_mul(table.heads)
                    .ok_or("Engram row count overflow")?
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (table.heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                upload.bytes.len() / 8
            };
            self.binding
                .shared
                .validate_start(i, self.binding.rank, sequence, rows)?;
        }
        Ok(())
    }

    pub fn enqueue(
        &self,
        sequence: u64,
        tokens: usize,
        uploads: &[Upload<'_>],
        inputs_ready: &Event,
    ) -> Result<()> {
        self.validate(sequence, tokens, uploads)?;
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let heads = table.heads;
            let requests = if let Some((address, flag)) = transfer.request {
                Requests::Device {
                    address,
                    flag,
                    sequence,
                    rows: tokens * heads,
                }
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                Requests::Host(
                    upload
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            };
            let (rows, generation) =
                self.binding
                    .shared
                    .start(i, self.binding.rank, sequence, requests)?;
            let frame = sequence as usize % STAGING_FRAMES;
            let staging = &self.binding.shared.layers[i].staging[frame];
            // The model stream records inputs_ready after the preceding graph, so DMA
            // cannot overwrite tables still consumed by that graph. Each graph waits for
            // its host gather before the following graph can overwrite the mailbox.
            transfer.stream.wait(inputs_ready)?;
            unsafe {
                transfer
                    .stream
                    .wait_value(transfer.flags[frame], generation)?;
                transfer.stream.upload(
                    transfer.weights,
                    &staging.data,
                    0,
                    rows * table.planes[0].row_bytes,
                )?;
                transfer.stream.upload(
                    transfer.scales,
                    &staging.data,
                    rows * table.planes[0].row_bytes,
                    rows * table.planes[1].row_bytes,
                )?;
                // The consumer is enqueued before a following submission re-records this event.
                transfer.stream.record_borrowed(transfer.event)?;
            }
        }
        Ok(())
    }
    /// See [`Shared::announce`].
    pub fn announce(&self, after: u64, windows: &[PromptWindow<'_>]) {
        self.binding
            .shared
            .announce(self.binding.rank, after, windows);
    }
    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }
    pub fn cancel(&self) {
        for layer in &self.binding.shared.layers {
            layer.feed.cancelled.store(true, Ordering::Release);
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut error = None;
        for transfer in &self.layers {
            error = error.or(transfer.stream.synchronize().err());
        }
        error.map_or(Ok(()), Err)
    }
}

pub(crate) fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Prefill
            | GraphKind::ScoredPrefill
            | GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Mixed
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_budget_covers_two_payloads_and_ready_flags_per_table() {
        let table = Table {
            planes: [Plane { file: 0, offset: 0, nbytes: 1024, row_bytes: 64 },
                Plane { file: 0, offset: 1024, nbytes: 128, row_bytes: 8 }],
            heads: 4, weights: "weights".into(), scales: "scales".into(),
            hashes: "hashes".into(), event: "ready".into(), request: None, hash: None,
        };
        let index = Index { tables: vec![table.clone(), table] };
        assert_eq!(staging_bytes(&index, 128).unwrap(), 2 * 2 * (128 * 4 * (64 + 8) + 4));
        assert!(staging_bytes(&index, usize::MAX).is_err());
    }

    #[test]
    fn two_frames_require_all_rank_retirement_before_reuse() {
        let mut states: [State; STAGING_FRAMES] = std::array::from_fn(|_| State::default());
        for sequence in 1..=2 {
            let state = &mut states[sequence as usize % STAGING_FRAMES];
            assert!(state.prepare(0, sequence, 24, 2).unwrap());
            assert!(!state.prepare(1, sequence, 24, 2).unwrap());
        }
        let first = &mut states[1];
        first.finish(0, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).is_err());
        assert_eq!(first.sequence, 1);
        first.finish(1, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).unwrap());
        assert_eq!(first.generation, 2);
        assert_eq!(first.rows, 48);
        assert_eq!(states[0].sequence, 2);
        assert_eq!(states[0].completed, 0);
    }

    #[test]
    fn frame_rejects_stale_duplicate_and_mismatched_rank_work() {
        let mut state = State::default();
        assert!(state.prepare(0, 0, 24, 2).is_err());
        assert!(state.prepare(2, 1, 24, 2).is_err());
        state.prepare(0, 1, 24, 2).unwrap();
        assert!(state.prepare(0, 1, 24, 2).is_err());
        assert!(state.prepare(1, 1, 48, 2).is_err());
        assert!(state.finish(1, 1).is_err());
        assert!(state.finish(0, 3).is_err());
        state.prepare(1, 1, 24, 2).unwrap();
        state.finish(0, 1).unwrap();
        assert!(state.finish(0, 1).is_err());
        state.finish(1, 1).unwrap();
        state.prepare(0, 3, 24, 2).unwrap();
        assert!(state.prepare(1, 1, 24, 2).is_err());
    }

    #[test]
    fn device_request_wait_observes_publication_and_cancellation() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        let cancelled = AtomicBool::new(false);
        let request = || Requests::Device {
            address: ids.as_ptr() as usize,
            flag: &flag as *const AtomicU64 as usize,
            sequence: 7,
            rows: ids.len(),
        };
        thread::scope(|scope| {
            let worker = scope.spawn(|| request().read(&cancelled));
            flag.store(7, Ordering::Release);
            assert_eq!(worker.join().unwrap().unwrap(), ids);
        });
        flag.store(0, Ordering::Release);
        cancelled.store(true, Ordering::Release);
        assert!(request().read(&cancelled).is_err());
    }

    /// Eight heads for each of the orders 2, 3 and 4. `first_rows` gives each head's first
    /// local row, or `None` for a head stored by another rank.
    fn row_hash(first_rows: impl Fn(usize, u64) -> Option<u64>) -> RowHash {
        const PRIMES: [u64; 24] = [
            101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
            191, 193, 197, 199, 211, 223, 227,
        ];
        let mut offset = 0;
        let heads = PRIMES
            .iter()
            .enumerate()
            .map(|(head, &prime)| {
                let first_row = first_rows(head, offset);
                offset += prime;
                Head { order: head / 8 + 2, prime, first_row }
            })
            .collect();
        RowHash {
            token_map: (0..16).map(|token| (token * 7 + 3) % 11).collect(),
            pad: 2,
            multipliers: vec![1000003, 998244353, 123456789, 987654321],
            heads,
        }
    }

    // Expected rows come from the reference equation of the captured hash,
    // `hash_reference` in b12x's `sequence/engram/reference.py`, evaluated over the
    // history that the DeepSeek V4.1 recipe builds for a prefill chunk.
    #[test]
    fn row_hash_matches_the_captured_graph_equation() {
        let hash = row_hash(|_, offset| Some(offset));
        assert_eq!(hash.history(), 3);
        // Token 20 in the history and token 99 in the chunk lie outside the vocabulary.
        let mut rows = Vec::new();
        hash.rows(&[5, 20, 7], &[1, 2, 99, 4, 15, 0], usize::MAX, &mut rows);
        let expected: [[i64; 24]; 6] = [
            [69, 152, 299, 402, 452, 573, 685, 908, 1009, 1162, 1223, 1408, 1561, 1745, 1956,
             2115, 2215, 2459, 2652, 2896, 3104, 3171, 3531, 3661],
            [64, 105, 275, 391, 499, 630, 744, 796, 976, 1129, 1276, 1482, 1592, 1810, 1988,
             2066, 2334, 2567, 2744, 2787, 3117, 3298, 3378, 3746],
            [82, 106, 285, 359, 466, 627, 665, 861, 999, 1107, 1268, 1494, 1597, 1711, 2016,
             2035, 2344, 2560, 2690, 2787, 2968, 3244, 3469, 3694],
            [88, 137, 228, 361, 438, 642, 774, 844, 985, 1120, 1244, 1483, 1676, 1833, 1952,
             2117, 2301, 2489, 2643, 2799, 3146, 3173, 3490, 3803],
            [5, 190, 216, 326, 484, 654, 675, 825, 1002, 1095, 1232, 1412, 1666, 1785, 1911,
             2033, 2379, 2543, 2726, 2848, 3109, 3366, 3415, 3611],
            [99, 189, 267, 367, 420, 632, 765, 918, 990, 1184, 1226, 1487, 1666, 1853, 1883,
             2136, 2351, 2412, 2640, 2934, 3077, 3259, 3520, 3683],
        ];
        assert_eq!(rows, expected.concat());

        // A longer history contributes only its last three tokens.
        let mut longer = Vec::new();
        hash.rows(&[8, 5, 20, 7], &[1, 2, 99, 4, 15, 0], usize::MAX, &mut longer);
        assert_eq!(longer, rows);

        // Positions before the prompt read as the padding token.
        rows.clear();
        hash.rows(&[9], &[3, 14], usize::MAX, &mut rows);
        let expected: [[i64; 24]; 2] = [
            [4, 156, 273, 385, 439, 543, 689, 871, 994, 1134, 1221, 1378, 1560, 1754, 1940,
             2117, 2345, 2466, 2682, 2882, 3022, 3172, 3447, 3706],
            [92, 203, 264, 397, 508, 604, 790, 903, 1012, 1200, 1300, 1436, 1674, 1729, 2001,
             2096, 2232, 2446, 2597, 2904, 3058, 3223, 3593, 3609],
        ];
        assert_eq!(rows, expected.concat());
        let mut padded = Vec::new();
        hash.rows(&[2, 2, 9], &[3, 14], usize::MAX, &mut padded);
        assert_eq!(padded, rows);
    }

    #[test]
    fn row_hash_reports_partition_local_rows_of_owned_heads_up_to_the_limit() {
        // Rank 1 of 3 stores heads 1, 4, 7, ... consecutively from its row zero.
        let mut cursor = 0;
        let mut local = [None; 24];
        let primes = row_hash(|_, _| None).heads;
        for head in (1..24).step_by(3) {
            local[head] = Some(cursor);
            cursor += primes[head].prime;
        }
        assert_eq!(cursor, 1272);
        let hash = row_hash(|head, _| local[head]);
        let mut rows = Vec::new();
        hash.rows(&[5, 20, 7], &[1, 2, 99, 4, 15, 0], usize::MAX, &mut rows);
        let expected: [[i64; 8]; 6] = [
            [51, 135, 333, 360, 562, 680, 977, 1202],
            [4, 182, 221, 413, 627, 799, 868, 1049],
            [5, 149, 286, 405, 528, 809, 868, 1140],
            [36, 121, 269, 381, 650, 766, 880, 1161],
            [89, 167, 250, 369, 602, 844, 929, 1086],
            [88, 103, 343, 363, 670, 816, 1015, 1191],
        ];
        assert_eq!(rows, expected.concat());

        let mut limited = vec![-7];
        hash.rows(&[5, 20, 7], &[1, 2, 99, 4, 15, 0], 12, &mut limited);
        assert_eq!(limited[0], -7);
        assert_eq!(limited[1..], rows[..11]);
        let mut none = Vec::new();
        row_hash(|_, _| None).rows(&[5, 20, 7], &[1, 2], usize::MAX, &mut none);
        assert!(none.is_empty());
    }

    /// A table file with one row per value of `row_hash`, whose primes sum to 3828, and a
    /// reader over it that gathers and reads ahead up to `step_tokens` tokens.
    #[cfg(unix)]
    fn hashed_table(
        name: &str,
        step_tokens: usize,
    ) -> (std::path::PathBuf, Vec<u8>, [Plane; 2], RowHash, reader::Reader, Arc<Stats>) {
        let hash = row_hash(|_, offset| Some(offset));
        let table_rows = 3828;
        let path = std::env::temp_dir().join(format!(
            "helios-engram-{name}-{}.bin",
            std::process::id()
        ));
        let bytes: Vec<u8> = (0..table_rows * 264)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane { file: 0, offset: 0, nbytes: table_rows * 256, row_bytes: 256 },
            Plane { file: 0, offset: (table_rows * 256) as u64, nbytes: table_rows * 8, row_bytes: 8 },
        ];
        let stats = Arc::new(Stats::default());
        let reader = reader::Reader::from_files(
            &[Arc::new(reader::open_file(&path).unwrap())],
            &planes,
            false,
            1 << 20,
            step_tokens * 24,
            stats.clone(),
            1,
            false,
            step_tokens * hash.local_heads(),
        )
        .unwrap();
        assert!(reader.reads_ahead());
        (path, bytes, planes, hash, reader, stats)
    }

    /// The rows a graph publishes for prompt windows, one per token and head, and how
    /// many of them are distinct.
    #[cfg(unix)]
    fn published(hash: &RowHash, windows: &[(Vec<u32>, Vec<u32>)]) -> (Vec<i64>, u64) {
        let mut ids = Vec::new();
        for (history, tokens) in windows {
            hash.rows(history, tokens, usize::MAX, &mut ids);
        }
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        (ids, unique.len() as u64)
    }

    /// Gather `ids` and compare every delivered byte with the table file.
    #[cfg(unix)]
    fn gather_checked(reader: &mut reader::Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
        let mut expected = Vec::new();
        for plane in planes {
            for &id in ids {
                let start = plane.offset as usize + id as usize * plane.row_bytes;
                expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
            }
        }
        let mut output = vec![0xa5; expected.len()];
        reader.gather(planes, ids, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    #[cfg(unix)]
    fn announced_prompt_windows_supply_the_rows_their_gather_requests() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let step_tokens = 8;
        let (path, bytes, planes, hash, mut reader, stats) = hashed_table("announced", step_tokens);
        let disk_reads = || stats.disk_reads.load(Ordering::Relaxed);

        // A reader with a gather waiting reads nothing ahead, and that gather reads from disk.
        let first = Announcement { after: 1, windows: vec![(vec![], vec![6, 1, 8, 13])] };
        read_ahead(&mut reader, &planes, &hash, &first, &|| true).unwrap();
        assert_eq!((disk_reads(), stats.prefetch_rows.load(Ordering::Relaxed)), (0, 0));
        let (ids, _) = published(&hash, &first.windows);
        gather_checked(&mut reader, &planes, &bytes, &ids);
        assert!(disk_reads() > 0);
        assert_eq!(stats.prefetch_hits.load(Ordering::Relaxed), 0);

        // Two prompts of the following step, eight tokens in all: every row the graph then
        // publishes is in memory, from the read-ahead or from the row cache.
        let second = Announcement {
            after: 2,
            windows: vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15]), (vec![9], vec![3, 14, 3])],
        };
        read_ahead(&mut reader, &planes, &hash, &second, &|| false).unwrap();
        let (ids, unique) = published(&hash, &second.windows);
        assert_eq!(ids.len(), step_tokens * 24);
        let (reads, hits) = (disk_reads(), stats.hits.load(Ordering::Relaxed));
        gather_checked(&mut reader, &planes, &bytes, &ids);
        assert_eq!(disk_reads(), reads);
        let cached = stats.hits.load(Ordering::Relaxed) - hits;
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), unique - cached);
        assert_eq!(stats.prefetch_hits.load(Ordering::Relaxed), unique - cached);
        std::fs::remove_file(path).unwrap();
    }

    // The scheduler chooses each step's width. The announcements below follow one prompt
    // by position, eight tokens past each submitted step, as the server derives them.
    #[test]
    #[cfg(unix)]
    fn steps_of_changing_width_read_each_predicted_row_once() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (path, bytes, planes, hash, mut reader, stats) = hashed_table("widths", 8);
        let prompt: [u32; 16] = [3, 14, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2];
        // The window of tokens `from..to` with its three tokens of history.
        let window = |from: usize, to: usize| (prompt[from - 3..from].to_vec(), prompt[from..to].to_vec());
        let counters = || {
            (
                stats.disk_reads.load(Ordering::Relaxed),
                stats.prefetch_rows.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
            )
        };

        // A step took tokens 3..5. Its announcement covers tokens 5..13: the next two, as
        // wide as that step, then the rest.
        let after_first = Announcement { after: 1, windows: vec![window(5, 7), window(7, 13)] };
        read_ahead(&mut reader, &planes, &hash, &after_first, &|| false).unwrap();
        let (_, ahead) = published(&hash, &[window(5, 13)]);
        let (reads, rows, _) = counters();
        assert_eq!(rows, ahead);

        // The following step is wider than the one before it, three tokens: its rows are
        // in memory, and the rows of tokens 8..13 stay held.
        let (ids, _) = published(&hash, &[window(5, 8)]);
        gather_checked(&mut reader, &planes, &bytes, &ids);
        assert_eq!(counters(), (reads, ahead, 0));

        // Its announcement covers tokens 8..16. Only rows that tokens 13..16 add are read.
        let after_second = Announcement { after: 2, windows: vec![window(8, 11), window(11, 16)] };
        read_ahead(&mut reader, &planes, &hash, &after_second, &|| false).unwrap();
        let (_, all) = published(&hash, &[window(5, 16)]);
        let (reads, rows, unused) = counters();
        assert_eq!((rows, unused), (all, 0));

        // A narrower step and then a full-width one find all their rows in memory.
        for (from, to) in [(8, 9), (9, 16)] {
            let (ids, _) = published(&hash, &[window(from, to)]);
            gather_checked(&mut reader, &planes, &bytes, &ids);
        }
        assert_eq!(counters(), (reads, all, 0));

        // A prediction for another prompt drops the rows held; all of them were used.
        let other = Announcement { after: 4, windows: vec![(vec![], vec![4, 4, 10])] };
        read_ahead(&mut reader, &planes, &hash, &other, &|| false).unwrap();
        let (_, added) = published(&hash, &other.windows);
        let (_, rows, unused) = counters();
        assert!(rows > all && rows <= all + added);
        assert_eq!(unused, 0);
        // Rows read ahead and never gathered are counted when a prediction drops them.
        read_ahead(&mut reader, &planes, &hash, &after_first, &|| false).unwrap();
        assert_eq!(counters().2, rows - all);
        std::fs::remove_file(path).unwrap();
    }

    // The scheduler alternates between prompts that prefill at once, four tokens a step.
    // Each announcement lists the prompt outside the submitted step first, as the server
    // orders them, and the read-ahead holds the rows of five tokens: one step and part of
    // the step listed after it.
    #[test]
    #[cfg(unix)]
    fn alternating_prompts_keep_their_rows_and_read_each_once() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (path, bytes, planes, hash, mut reader, stats) = hashed_table("alternate", 5);
        let first: [u32; 19] = [3, 14, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4];
        let second: [u32; 19] = [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5];
        // The four tokens of `prompt` from `from`, with their three tokens of history.
        let window = |prompt: &[u32; 19], from: usize| {
            (prompt[from - 3..from].to_vec(), prompt[from..from + 4].to_vec())
        };
        let counters = || {
            (
                stats.disk_reads.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
            )
        };
        let step = |reader: &mut reader::Reader, prompt: &[u32; 19], from: usize| {
            let (ids, _) = published(&hash, &[window(prompt, from)]);
            gather_checked(reader, &planes, &bytes, &ids);
        };
        let announce = |reader: &mut reader::Reader, after: u64, windows: Vec<(Vec<u32>, Vec<u32>)>| {
            let announcement = Announcement { after, windows };
            read_ahead(reader, &planes, &hash, &announcement, &|| false).unwrap();
        };

        // The first prompt's step runs; the second prompt is expected next.
        step(&mut reader, &first, 3);
        announce(&mut reader, 1, vec![window(&second, 3), window(&first, 7)]);
        // Each step that follows finds its rows in memory, and each announcement reads
        // the rows of the prompt expected after it.
        let (reads, _) = counters();
        step(&mut reader, &second, 3);
        assert_eq!(counters(), (reads, 0));
        announce(&mut reader, 2, vec![window(&first, 7), window(&second, 7)]);
        let (reads, _) = counters();
        step(&mut reader, &first, 7);
        assert_eq!(counters(), (reads, 0));
        announce(&mut reader, 3, vec![window(&second, 7), window(&first, 11)]);

        // The scheduler runs the first prompt again instead. Its rows are read by the
        // gather; the rows held for the second prompt are still listed, stay, and serve
        // its step when it comes.
        step(&mut reader, &first, 11);
        announce(&mut reader, 4, vec![window(&second, 7), window(&first, 15)]);
        let (reads, _) = counters();
        step(&mut reader, &second, 7);
        assert_eq!(counters(), (reads, 0));
        std::fs::remove_file(path).unwrap();
    }

    // Two prompts alternate in steps of two tokens, a quarter of the widest graph, whose
    // rows the read-ahead holds. Each announcement lists, as the server does, the next
    // step of the prompt expected next and of the prompt just served, then the rest of
    // each prompt's eight tokens. The rows held for one prompt's rest must not make room
    // for the other's: every step finds its rows in memory, and no row is dropped unused.
    #[test]
    #[cfg(unix)]
    fn narrow_alternating_steps_keep_the_rows_held_for_each_prompt() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (path, bytes, planes, hash, mut reader, stats) = hashed_table("narrow", 8);
        let prompts: [[u32; 27]; 2] = [
            [3, 14, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3],
            [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3, 6, 0, 2, 8, 7, 4, 7],
        ];
        // Tokens `from..to` of a prompt, cut at its end, with their three tokens of history.
        let window = |prompt: usize, from: usize, to: usize| {
            let prompt = &prompts[prompt];
            (prompt[from - 3..from].to_vec(), prompt[from..to.min(prompt.len())].to_vec())
        };
        let counters = || {
            (
                stats.disk_reads.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
            )
        };
        // Positions reached by the two prompts.
        let mut at = [3, 3];
        let mut sequence = 0;
        // Run the next step of `served`, then announce with the other prompt expected next.
        let mut step = |reader: &mut reader::Reader, served: usize, held: bool| {
            let (reads, _) = counters();
            let (ids, _) = published(&hash, &[window(served, at[served], at[served] + 2)]);
            gather_checked(reader, &planes, &bytes, &ids);
            if held {
                assert_eq!(counters(), (reads, 0), "prompt {served} at {}", at[served]);
            }
            at[served] += 2;
            sequence += 1;
            let order = [1 - served, served];
            let windows = order
                .iter()
                .map(|&prompt| window(prompt, at[prompt], at[prompt] + 2))
                .chain(order.iter().map(|&prompt| window(prompt, at[prompt] + 2, at[prompt] + 8)))
                .collect();
            let announcement = Announcement { after: sequence, windows };
            read_ahead(reader, &planes, &hash, &announcement, &|| false).unwrap();
        };
        // The first step's rows were not announced; every later step's were.
        step(&mut reader, 0, false);
        for _ in 0..6 {
            step(&mut reader, 1, true);
            step(&mut reader, 0, true);
        }
        let read = stats.prefetch_rows.load(Ordering::Relaxed);
        let used = stats.prefetch_hits.load(Ordering::Relaxed);
        // The rows read and not yet used are held for the steps that would follow.
        assert!(used > 0 && read >= used && read - used <= 8 * 24, "{read} read, {used} used");
        assert_eq!(counters().1, 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn announcement_waits_for_its_gather_and_for_gathers_in_flight() {
        let announcement = |after| Some(Arc::new(Announcement { after, windows: Vec::new() }));
        // Announced before the gather it follows was received: it stays pending.
        let mut announced = announcement(7);
        assert!(matches!(due(&mut announced, 6, false), Due::Idle));
        assert!(announced.is_some());
        // The gather for sequence 7 has been served, but another gather of the group is
        // in flight: the announcement stays, to be asked for again.
        assert!(matches!(due(&mut announced, 7, true), Due::Wait));
        assert!(announced.is_some());
        // No gather is in flight: the read-ahead runs once.
        assert!(matches!(due(&mut announced, 7, false), Due::Read(taken) if taken.after == 7));
        assert!(announced.is_none());
        assert!(matches!(due(&mut announced, 7, false), Due::Idle));
        // A later gather started first: the announcement expires.
        for gathering in [false, true] {
            let mut announced = announcement(7);
            assert!(matches!(due(&mut announced, 8, gathering), Due::Idle));
            assert!(announced.is_none());
        }
    }

    #[test]
    fn only_the_first_rank_of_a_group_announces_and_the_listing_ends_at_its_extent() {
        let windows = [
            PromptWindow { history: &[], tokens: &[1, 2, 3] },
            PromptWindow { history: &[9], tokens: &[] },
            PromptWindow { history: &[3, 4], tokens: &[5, 6, 7] },
            PromptWindow { history: &[7], tokens: &[8] },
        ];
        // The ranks of a delivery group share its readers: the first rank's call serves all.
        assert!(announcement(1, 4, &windows, 16).is_none());
        // Windows keep their order and their history; one without tokens lists nothing.
        let whole = announcement(0, 4, &windows, 16).unwrap();
        assert_eq!(whole.after, 4);
        assert_eq!(whole.windows, [
            (vec![], vec![1, 2, 3]),
            (vec![3, 4], vec![5, 6, 7]),
            (vec![7], vec![8]),
        ]);
        // The listing ends after `listed` tokens, inside a window if need be.
        let cut = announcement(0, 4, &windows, 5).unwrap();
        assert_eq!(cut.windows, [(vec![], vec![1, 2, 3]), (vec![3, 4], vec![5, 6])]);
        assert_eq!(announcement(0, 4, &windows, 3).unwrap().windows, [(vec![], vec![1, 2, 3])]);
        // Nothing to list makes no announcement.
        assert!(announcement(0, 4, &windows, 0).is_none());
        assert!(announcement(0, 4, &windows[1..2], 16).is_none());
        assert!(announcement(0, 4, &[], 16).is_none());
    }

    #[test]
    fn the_announcement_extent_follows_the_table_catalog() {
        let table = |hash: Option<RowHash>| Table {
            planes: [256, 8].map(|row_bytes| Plane {
                file: 0,
                offset: 0,
                nbytes: 100 * row_bytes,
                row_bytes,
            }),
            heads: 24,
            weights: "weights".into(),
            scales: "scales".into(),
            hashes: "hashes".into(),
            event: "ready".into(),
            request: Some("requests".into()),
            hash,
        };
        let extent = |tables: Vec<Table>| Shared::layout_fixture(Index { tables }, 1).read_ahead_extent();
        // No table whose rows are derived from prompt tokens: nothing to announce.
        assert_eq!(extent(vec![]), None);
        assert_eq!(extent(vec![table(None)]), None);
        // The longest n-gram sets the history; a step of the widest graph, 4096 tokens in
        // this fixture, the tokens; and `LISTED_STEPS` steps the listing.
        let hash = row_hash(|_, offset| Some(offset));
        let mut shorter = hash.clone();
        shorter.multipliers.truncate(2);
        assert_eq!((hash.history(), shorter.history()), (3, 1));
        let listed = 4096 * LISTED_STEPS;
        assert_eq!(extent(vec![table(Some(shorter.clone()))]), Some((1, 4096, listed)));
        assert_eq!(
            extent(vec![table(Some(shorter)), table(None), table(Some(hash))]),
            Some((3, 4096, listed))
        );
        // The extent does not depend on this group's readers, which a fixture lacks.
        assert!(!Shared::layout_fixture(Index { tables: vec![] }, 1).reads_ahead());
    }

    /// Host-memory frames that record what a reader publishes.
    #[cfg(unix)]
    struct HostFrames {
        rows: [std::cell::UnsafeCell<Vec<u8>>; STAGING_FRAMES],
        signals: Mutex<mpsc::Sender<(usize, u32)>>,
        failures: Mutex<Vec<(usize, String)>>,
    }
    // SAFETY: The reader writes a frame only before signalling it, and a test reads a frame
    // only after receiving that signal and before submitting the frame again.
    #[cfg(unix)]
    unsafe impl Sync for HostFrames {}
    #[cfg(unix)]
    impl Frames for HostFrames {
        unsafe fn rows(&self, frame: usize, nbytes: usize) -> &mut [u8] {
            // SAFETY: See the `Sync` implementation.
            unsafe { &mut (&mut *self.rows[frame].get())[..nbytes] }
        }
        fn signal(&self, frame: usize, generation: u32) {
            let _ = self.signals.lock().unwrap().send((frame, generation));
        }
        fn fail(&self, frame: usize, error: Error) {
            self.failures.lock().unwrap().push((frame, error.to_string()));
        }
    }

    /// A reader thread over `hashed_table`, delivering into host memory.
    #[cfg(unix)]
    struct ReaderThread {
        feed: Feed,
        gathering: Arc<AtomicUsize>,
        frames: Arc<HostFrames>,
        signals: mpsc::Receiver<(usize, u32)>,
        thread: Option<JoinHandle<()>>,
        path: std::path::PathBuf,
        bytes: Vec<u8>,
        planes: [Plane; 2],
        hash: RowHash,
        stats: Arc<Stats>,
    }
    #[cfg(unix)]
    impl ReaderThread {
        const STEP_TOKENS: usize = 8;

        fn start(name: &str) -> Self {
            Self::deriving(name, |hash| hash)
        }

        /// A reader thread whose row derivation is `derivation` of the table's.
        fn deriving(name: &str, derivation: impl FnOnce(RowHash) -> RowHash) -> Self {
            Self::prepared(name, derivation, |_| {})
        }

        /// [`Self::deriving`], with `prepare` applied to the reader before its thread starts.
        fn prepared(
            name: &str,
            derivation: impl FnOnce(RowHash) -> RowHash,
            prepare: impl FnOnce(&mut reader::Reader),
        ) -> Self {
            let (path, bytes, planes, hash, mut reader, stats) =
                hashed_table(name, Self::STEP_TOKENS);
            prepare(&mut reader);
            let (sender, signals) = mpsc::channel();
            let frames = Arc::new(HostFrames {
                rows: std::array::from_fn(|_| {
                    std::cell::UnsafeCell::new(vec![0xa5; Self::STEP_TOKENS * 24 * 264])
                }),
                signals: Mutex::new(sender),
                failures: Mutex::new(Vec::new()),
            });
            let gathering = Arc::new(AtomicUsize::new(0));
            let (feed, source) = link(&gathering, true);
            let (delivery, derivation) = (frames.clone(), derivation(hash.clone()));
            let thread = thread::spawn(move || {
                serve(source, reader, planes, Some(derivation), &*delivery)
            });
            Self { feed, gathering, frames, signals, thread: Some(thread), path, bytes, planes, hash, stats }
        }

        /// Submit the gather of `ids` as submission `sequence` and return the frame's
        /// rows once the reader signals them.
        fn gather(&self, sequence: u64, ids: &[i64]) -> Vec<u8> {
            let frame = sequence as usize % STAGING_FRAMES;
            self.feed
                .gather(Job {
                    frame,
                    generation: sequence as u32,
                    sequence,
                    requests: Requests::Host(ids.to_vec()),
                })
                .unwrap();
            let signal = self.signals.recv_timeout(Duration::from_secs(10)).unwrap();
            assert_eq!(signal, (frame, sequence as u32));
            // SAFETY: The frame was signalled and is not submitted again before this copy.
            unsafe { (&*self.frames.rows[frame].get())[..ids.len() * 264].to_vec() }
        }

        /// The bytes a gather of `ids` must deliver.
        fn expected(&self, ids: &[i64]) -> Vec<u8> {
            let mut expected = Vec::new();
            for plane in &self.planes {
                for &id in ids {
                    let start = plane.offset as usize + id as usize * plane.row_bytes;
                    expected.extend_from_slice(&self.bytes[start..start + plane.row_bytes]);
                }
            }
            expected
        }

        fn announce(&self, after: u64, windows: Vec<(Vec<u32>, Vec<u32>)>) {
            self.feed.announce(&Arc::new(Announcement { after, windows }));
        }

        /// Wait until the reader thread reaches `condition`.
        fn wait_for(&self, what: &str, condition: impl Fn(&Self) -> bool) {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !condition(self) {
                assert!(std::time::Instant::now() < deadline, "reader thread did not reach: {what}");
                thread::sleep(Duration::from_millis(1));
            }
        }

        /// Read-ahead passes completed or abandoned so far, by their accumulated time.
        fn passes_ns(&self) -> u64 {
            self.stats.prefetch_ns.load(Ordering::Relaxed)
        }

        /// Drop the feed's sender, as dropping its layer does, and join the reader thread.
        fn stop(mut self) {
            self.feed.sender.take();
            self.thread.take().unwrap().join().unwrap();
            std::fs::remove_file(&self.path).unwrap();
        }
    }

    #[test]
    #[cfg(unix)]
    fn reader_thread_reads_ahead_after_the_announced_gather_and_serves_the_next() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let table = ReaderThread::start("thread");
        let disk_reads = || table.stats.disk_reads.load(Ordering::Relaxed);
        let following = vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15]), (vec![9], vec![3, 14, 3])];

        // The announcement arrives before the gather it follows, as it does on a rank
        // whose commit comes later: the reader takes its notice and reads nothing yet.
        table.announce(1, following.clone());
        table.wait_for("the notice", |table| !table.feed.notified.load(Ordering::Acquire));
        assert!(table.feed.announced.lock().unwrap().is_some());
        assert_eq!((disk_reads(), table.passes_ns()), (0, 0));

        // The gather is delivered, is no longer in flight, and the read-ahead follows it.
        let (ids, _) = published(&table.hash, &[(vec![], vec![6, 1, 8, 13])]);
        assert_eq!(table.gather(1, &ids), table.expected(&ids));
        table.wait_for("the read-ahead", |table| table.passes_ns() != 0);
        assert_eq!(table.gathering.load(Ordering::Acquire), 0);
        assert!(table.feed.announced.lock().unwrap().is_none());
        assert!(table.stats.prefetch_rows.load(Ordering::Relaxed) > 0);

        // The following gather finds every row in memory.
        let (ids, _) = published(&table.hash, &following);
        let reads = disk_reads();
        assert_eq!(table.gather(2, &ids), table.expected(&ids));
        assert_eq!(disk_reads(), reads);
        assert!(table.stats.prefetch_hits.load(Ordering::Relaxed) > 0);

        // Announcements made faster than the reader takes them never fill its channel:
        // the latest one stays, and one made for a gather already passed is discarded.
        for _ in 0..64 {
            table.announce(1, following.clone());
        }
        table.wait_for("the stale notices", |table| {
            !table.feed.notified.load(Ordering::Acquire)
                && table.feed.announced.lock().unwrap().is_none()
        });
        assert!(table.frames.failures.lock().unwrap().is_empty());
        table.stop();
    }

    #[test]
    #[cfg(unix)]
    fn reader_thread_leaves_the_disk_to_gathers_in_flight_for_other_tables() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let table = ReaderThread::start("yielding");
        let (ids, _) = published(&table.hash, &[(vec![], vec![6, 1, 8, 13])]);
        assert_eq!(table.gather(1, &ids), table.expected(&ids));
        table.wait_for("the delivery", |table| table.gathering.load(Ordering::Acquire) == 0);

        // Another table of the group has a gather in flight: the announcement is due and
        // waits, without reading.
        table.gathering.fetch_add(1, Ordering::AcqRel);
        let reads = table.stats.disk_reads.load(Ordering::Relaxed);
        table.announce(1, vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15])]);
        table.wait_for("the notice", |table| !table.feed.notified.load(Ordering::Acquire));
        thread::sleep(Duration::from_millis(20));
        assert!(table.feed.announced.lock().unwrap().is_some());
        assert_eq!(table.stats.disk_reads.load(Ordering::Relaxed), reads);
        assert_eq!(table.passes_ns(), 0);

        // That gather is delivered: the read-ahead runs without another notice.
        table.gathering.fetch_sub(1, Ordering::AcqRel);
        table.wait_for("the read-ahead", |table| table.passes_ns() != 0);
        assert!(table.stats.disk_reads.load(Ordering::Relaxed) > reads);
        assert!(table.stats.prefetch_rows.load(Ordering::Relaxed) > 0);
        assert_eq!(table.stats.prefetch_interrupted.load(Ordering::Relaxed), 0);
        table.stop();
    }

    #[test]
    #[cfg(unix)]
    fn a_failed_read_ahead_read_is_left_to_the_gather() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let table = ReaderThread::start("failing");
        let (ids, _) = published(&table.hash, &[(vec![], vec![6, 1, 8, 13])]);
        assert_eq!(table.gather(1, &ids), table.expected(&ids));

        // The table file loses its rows: reading ahead rows not yet cached fails once
        // its I/O has completed. The pass ends, and nothing else changes.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&table.path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        let following = vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15])];
        table.announce(1, following.clone());
        table.wait_for("the failed read", |table| {
            table.stats.prefetch_errors.load(Ordering::Relaxed) == 1
        });
        assert!(!table.feed.cancelled.load(Ordering::Acquire));
        assert!(table.feed.failure.lock().unwrap().is_none());
        assert!(table.frames.failures.lock().unwrap().is_empty());

        // With its rows back, the gather of those rows reads them itself.
        std::fs::write(&table.path, &table.bytes).unwrap();
        let (ids, _) = published(&table.hash, &following);
        let reads = table.stats.disk_reads.load(Ordering::Relaxed);
        assert_eq!(table.gather(2, &ids), table.expected(&ids));
        assert!(table.stats.disk_reads.load(Ordering::Relaxed) > reads);
        table.stop();
    }

    /// A derivation that panics on its first head.
    #[cfg(unix)]
    fn panicking(mut hash: RowHash) -> RowHash {
        hash.heads[0].order = hash.multipliers.len() + 1;
        hash
    }

    #[test]
    #[cfg(unix)]
    fn a_panic_while_reading_ahead_stops_read_ahead_and_keeps_the_table_serving() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let table = ReaderThread::deriving("panic", panicking);
        let (ids, _) = published(&table.hash, &[(vec![], vec![6, 1, 8, 13])]);
        assert_eq!(table.gather(1, &ids), table.expected(&ids));
        assert!(table.stats.prefetch_capacity_rows.load(Ordering::Relaxed) > 0);
        table.announce(1, vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15])]);
        table.wait_for("the panic", |table| {
            table.stats.prefetch_errors.load(Ordering::Relaxed) == 1
        });
        // Every read of the reader had completed, so the table gives up read-ahead and
        // nothing fails: no error waits for a submission, and no frame reports one.
        assert_eq!(table.stats.prefetch_capacity_rows.load(Ordering::Relaxed), 0);
        assert!(table.feed.failure.lock().unwrap().is_none());
        assert!(!table.feed.cancelled.load(Ordering::Acquire));
        // Gathers are served from disk as before, and a later announcement reads nothing:
        // the reader takes it before the gather that follows it.
        let (ids, _) = published(&table.hash, &[(vec![6, 1, 8], vec![13, 3, 2, 7])]);
        assert_eq!(table.gather(2, &ids), table.expected(&ids));
        table.announce(2, vec![(vec![3, 2, 7], vec![9, 4, 1, 5])]);
        let (ids, _) = published(&table.hash, &[(vec![3, 2, 7], vec![9, 4, 1, 5])]);
        let reads = table.stats.disk_reads.load(Ordering::Relaxed);
        assert_eq!(table.gather(3, &ids), table.expected(&ids));
        assert!(table.stats.disk_reads.load(Ordering::Relaxed) > reads);
        assert_eq!(table.stats.prefetch_rows.load(Ordering::Relaxed), 0);
        assert!(table.frames.failures.lock().unwrap().is_empty());
        table.stop();
    }

    #[test]
    #[cfg(unix)]
    fn a_reader_left_unusable_by_read_ahead_fails_the_table_not_a_frame() {
        let _one_reader = reader::READ_AHEAD_READER
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // The reader has reads outstanding when its read-ahead pass ends in a panic, as
        // after a failure of the pass's own reads: it cannot serve gathers either.
        let table = ReaderThread::prepared("unusable", panicking, |reader| {
            reader.leave_reads_outstanding()
        });
        let (ids, _) = published(&table.hash, &[(vec![], vec![6, 1, 8, 13])]);
        table.announce(0, vec![(vec![5, 20, 7], vec![1, 2, 99, 4, 15])]);
        table.wait_for("the failure", |table| table.feed.cancelled.load(Ordering::Acquire));

        // The failure waits in the table's own slot for the next submission. No frame
        // reports it: no step had rows outstanding.
        let failure = || table.feed.failure.lock().unwrap().as_ref().map(|error| error.to_string());
        assert!(failure().is_some_and(|error| error.contains("panicked")));
        assert!(table.frames.failures.lock().unwrap().is_empty());

        // Later announcements are not taken. A gather that reaches the reader nonetheless,
        // as one queued during the failing pass does, is signalled with zeroed rows and
        // the table's failure, so that its graph can drain and its step reports the cause.
        table.announce(0, vec![(vec![], vec![4])]);
        assert!(!table.feed.notified.load(Ordering::Acquire));
        assert!(table.gather(1, &ids).iter().all(|byte| *byte == 0));
        table.wait_for("the delivery", |table| table.gathering.load(Ordering::Acquire) == 0);
        {
            let failures = table.frames.failures.lock().unwrap();
            assert_eq!((failures.len(), failures[0].0), (1, 1));
            assert!(failures[0].1.contains("panicked"), "{}", failures[0].1);
        }
        assert!(failure().is_some_and(|error| error.contains("panicked")));
        table.stop();
    }
}
