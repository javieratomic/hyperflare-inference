/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use super::{Stats, cache::RowCache};
use crate::Result;
#[cfg(unix)]
use io_uring::{IoUring, opcode, types};
use std::{
    fs::File,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

#[cfg(test)]
use std::path::Path;

#[cfg(windows)]
mod windows;

const BLOCK: usize = 4096;
const CHUNK: usize = 64 << 10;
const DEPTH: usize = 128;
const LOAD_CHUNK: usize = 4 << 20;
const LOAD_READERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plane {
    pub file: usize,
    pub offset: u64,
    pub nbytes: usize,
    pub row_bytes: usize,
}

impl Plane {
    pub(crate) fn resident_bytes(self) -> Result<usize> {
        ((self.offset % BLOCK as u64) as usize)
            .checked_add(self.nbytes)
            .and_then(|n| n.checked_add(BLOCK - 1))
            .map(|n| (n / BLOCK * BLOCK).max(BLOCK))
            .ok_or_else(|| "resident embedding allocation exceeds host range".into())
    }
}

#[derive(Clone, Copy)]
struct Fragment {
    file: usize,
    source: u64,
    destination: usize,
    nbytes: usize,
}

struct Read {
    file: usize,
    offset: u64,
    nbytes: usize,
    required: usize,
    fragments: std::ops::Range<usize>,
}

struct Aligned(usize, usize);
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self(
            crate::os::allocate_aligned(nbytes).map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        ))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}

/// Each reader locks 8 MiB of io_uring staging. The read-ahead tests hold one reader at a
/// time, so that together they add one reader to the test binary's locked memory.
#[cfg(test)]
pub(super) static READ_AHEAD_READER: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(super) fn open_file(path: &Path) -> Result<File> {
    crate::os::direct_file(path, false).map_err(|e| format!("open Engram core: {e}").into())
}

pub(crate) struct Resident {
    #[cfg(unix)]
    storage: ResidentMapping,
    #[cfg(windows)]
    storage: Aligned,
    #[cfg(unix)]
    file: File,
    skip: usize,
    nbytes: usize,
}

#[cfg(unix)]
struct ResidentMapping {
    address: usize,
    nbytes: usize,
}

#[cfg(unix)]
impl ResidentMapping {
    fn new(file: &File, nbytes: usize, writable: bool) -> Result<Self> {
        let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                protection,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        let mapping = Self {
            address: address as usize,
            nbytes,
        };
        // The sealed read-only remap has its own page tables, even after loading
        // populated the backing RAM. Prepare them before row delivery starts.
        if !writable && unsafe { libc::madvise(address, nbytes, libc::MADV_POPULATE_READ) } != 0 {
            let error = std::io::Error::last_os_error();
            return Err(format!("populate resident embedding page tables: {error}").into());
        }
        Ok(mapping)
    }
}

#[cfg(unix)]
impl Drop for ResidentMapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.address as *mut libc::c_void, self.nbytes) };
    }
}

#[cfg(unix)]
const RESIDENT_SEALS: i32 =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[cfg(unix)]
fn seal_resident(file: &File) -> Result<()> {
    // The loading threads have joined and their writable mapping is gone.
    // Kernel-held page references can still make F_SEAL_WRITE temporarily busy.
    // Publish only after every required seal succeeds; persistent busy state fails.
    for attempt in 0..4 {
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EBUSY) || attempt == 3 {
            return Err(format!("seal resident embedding table: {error}").into());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unreachable!()
}

impl Resident {
    pub(super) fn load(file: &File, plane: Plane) -> Result<Self> {
        Self::initialize(plane, |address, extent, required| {
            let skip = (plane.offset % BLOCK as u64) as usize;
            let offset = plane.offset - skip as u64;
            std::thread::scope(|scope| -> Result<()> {
                let mut workers = Vec::new();
                for reader in 0..LOAD_READERS.min(extent.div_ceil(LOAD_CHUNK)) {
                    workers.push(
                        std::thread::Builder::new()
                            .name(format!("flare-table-loader-{reader}"))
                            .spawn_scoped(scope, move || -> Result<()> {
                                for begin in (reader * LOAD_CHUNK..extent)
                                    .step_by(LOAD_READERS * LOAD_CHUNK)
                                {
                                    let nbytes = LOAD_CHUNK.min(extent - begin);
                                    let source = offset
                                        .checked_add(begin as u64)
                                        .ok_or("resident Engram file offset exceeds range")?;
                                    // SAFETY: Each reader initializes a disjoint aligned allocation range.
                                    let count = unsafe { crate::os::read_at_raw(file, (address + begin) as *mut u8, nbytes, source) }
                                        .map_err(|e| format!("resident Engram read at {source}: {e}"))?;
                                    let needed = nbytes.min(required - begin);
                                    if count < needed {
                                        return Err(format!(
                                            "resident Engram read at {source} returned {count}, requires {needed} bytes"
                                        ).into());
                                    }
                                }
                                Ok(())
                            })
                            .map_err(|e| format!("start resident Engram loader: {e}"))?,
                    );
                }
                for worker in workers {
                    worker
                        .join()
                        .map_err(|_| "resident Engram loader panicked")??;
                }
                Ok(())
            })
        })
    }

    #[cfg(unix)]
    pub(super) fn load_segments(plane: Plane, segments: &[(File, u64, usize)]) -> Result<Self> {
        if plane.offset != 0 {
            return Err("assembled embedding planes must start at offset zero".into());
        }
        Self::initialize(plane, |address, _, _| {
            let mut destination = 0usize;
            for (file, offset, nbytes) in segments {
                let end = destination
                    .checked_add(*nbytes)
                    .filter(|end| *end <= plane.nbytes)
                    .ok_or("embedding source segments exceed their plane")?;
                let mut copied = 0;
                while copied < *nbytes {
                    let count = (*nbytes - copied).min(LOAD_CHUNK);
                    // SAFETY: Source segments initialize consecutive, disjoint ranges
                    // inside the resident allocation before it becomes immutable.
                    let read = unsafe {
                        crate::os::read_at_raw(
                            file,
                            (address + destination + copied) as *mut u8,
                            count,
                            offset + copied as u64,
                        )
                    }
                    .map_err(|error| format!("read embedding source segment: {error}"))?;
                    if read == 0 {
                        return Err(
                            "embedding source segment ended before its declared size".into()
                        );
                    }
                    copied += read;
                }
                destination = end;
            }
            if destination != plane.nbytes {
                return Err("embedding source segments do not fill their plane".into());
            }
            Ok(())
        })
    }

    fn initialize(
        plane: Plane,
        fill: impl FnOnce(usize, usize, usize) -> Result<()>,
    ) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = plane.resident_bytes()?;
        #[cfg(unix)]
        let backing = {
            let descriptor = unsafe {
                libc::memfd_create(
                    c"flare-embedding-table".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                )
            };
            if descriptor < 0 {
                return Err(format!(
                    "create resident embedding table: {}",
                    std::io::Error::last_os_error()
                )
                .into());
            }
            // The descriptor is newly created and owned exclusively until published.
            let backing = unsafe { File::from_raw_fd(descriptor) };
            backing
                .set_len(extent as u64)
                .map_err(|e| format!("size resident embedding table: {e}"))?;
            backing
        };
        #[cfg(unix)]
        let storage = ResidentMapping::new(&backing, extent, true)?;
        #[cfg(unix)]
        let address = storage.address;
        #[cfg(windows)]
        let storage = Aligned::new(extent)?;
        #[cfg(windows)]
        let address = storage.0;
        fill(address, extent, required)?;
        #[cfg(unix)]
        {
            drop(storage);
            seal_resident(&backing)?;
            Self::from_shared(backing, plane)
        }
        #[cfg(windows)]
        {
            Ok(Self {
                storage,
                skip,
                nbytes: plane.nbytes,
            })
        }
    }

    #[cfg(unix)]
    pub(super) fn from_shared(file: File, plane: Plane) -> Result<Self> {
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & RESIDENT_SEALS != RESIDENT_SEALS {
            return Err("resident embedding table must have immutable size and contents".into());
        }
        let skip = (plane.offset % BLOCK as u64) as usize;
        let extent = plane.resident_bytes()?;
        if file
            .metadata()
            .map_err(|e| format!("inspect resident embedding table: {e}"))?
            .len()
            != extent as u64
        {
            return Err("shared resident embedding table extent differs from its plane".into());
        }
        Ok(Self {
            storage: ResidentMapping::new(&file, extent, false)?,
            file,
            skip,
            nbytes: plane.nbytes,
        })
    }

    #[cfg(unix)]
    pub(super) fn descriptor(&self) -> i32 {
        self.file.as_raw_fd()
    }

    fn bytes(&self) -> &[u8] {
        #[cfg(unix)]
        let address = self.storage.address;
        #[cfg(windows)]
        let address = self.storage.0;
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe { std::slice::from_raw_parts((address + self.skip) as *const u8, self.nbytes) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { crate::os::free_aligned(self.0, self.1) };
    }
}

pub(crate) struct DiskReader {
    // Destroying the ring cancels and drains outstanding I/O before the buffers are freed.
    #[cfg(unix)]
    ring: IoUring,
    #[cfg(windows)]
    windows: windows::Readers,
    #[cfg(unix)]
    staging: Aligned,
    #[cfg(unix)]
    files: Vec<Arc<File>>,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
    resident_scales: Option<Resident>,
    /// Whether every read handed to the kernel has completed. A failure that leaves reads
    /// outstanding makes the reader unusable.
    settled: bool,
    stats: Arc<Stats>,
}

// Accumulate locally so timing does not add atomic operations to each completion.
// Drop also records completed phases when a gather exits with an I/O error.
struct DiskTimings<'a> {
    stats: &'a Stats,
    plan: u64,
    submit: u64,
    wait: u64,
    complete: u64,
}

impl Drop for DiskTimings<'_> {
    fn drop(&mut self) {
        for (counter, value) in [
            (&self.stats.disk_plan_ns, self.plan),
            (&self.stats.disk_submit_ns, self.submit),
            (&self.stats.disk_wait_ns, self.wait),
            (&self.stats.disk_complete_ns, self.complete),
        ] {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }
}

impl DiskReader {
    fn from_files(files: &[Arc<File>], stats: Arc<Stats>, workers: usize) -> Result<Self> {
        #[cfg(unix)]
        if workers != 1 {
            return Err("parallel embedding disk workers require Windows".into());
        }
        #[cfg(unix)]
        let staging = Aligned::new(DEPTH * CHUNK)?;
        #[cfg(unix)]
        let ring = {
            let ring = IoUring::new(DEPTH as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
            let buffers: Vec<_> = (0..DEPTH)
                .map(|i| libc::iovec {
                    iov_base: staging.at(i).cast(),
                    iov_len: CHUNK,
                })
                .collect();
            unsafe { ring.submitter().register_buffers(&buffers) }
                .map_err(|e| format!("register Engram staging: {e}"))?;
            ring
        };
        #[cfg(windows)]
        let windows = windows::Readers::new(files, workers)?;
        Ok(Self {
            #[cfg(windows)]
            windows,
            #[cfg(unix)]
            ring,
            #[cfg(unix)]
            staging,
            #[cfg(unix)]
            files: files.to_vec(),
            fragments: Vec::new(),
            reads: Vec::new(),
            resident_scales: None,
            settled: true,
            stats,
        })
    }

    /// Rows whose reads normally fit one batch: a row takes one read per plane read from
    /// disk unless a read reaches its size limit between the row's two blocks.
    fn batch_rows(&self) -> usize {
        if self.resident_scales.is_some() {
            DEPTH
        } else {
            DEPTH / 2
        }
    }

    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        let mut timing = DiskTimings {
            stats: &self.stats,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        let begin = Instant::now();
        self.fragments.clear();
        let mut base = 0;
        for (ordinal, plane) in planes.iter().enumerate() {
            if ordinal == 1 {
                if let Some(scales) = &self.resident_scales {
                    gather_resident(scales, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                    continue;
                }
            }
            for (row, id) in ids.iter().copied().enumerate() {
                let destination = base + row * plane.row_bytes;
                if id < 0 {
                    output[destination..destination + plane.row_bytes].fill(0);
                    continue;
                }
                let source = plane.offset + id as u64 * plane.row_bytes as u64;
                let mut copied = 0;
                while copied < plane.row_bytes {
                    let address = source + copied as u64;
                    let nbytes = (plane.row_bytes - copied).min(BLOCK - address as usize % BLOCK);
                    self.fragments.push(Fragment {
                        file: plane.file,
                        source: address,
                        destination: destination + copied,
                        nbytes,
                    });
                    copied += nbytes;
                }
            }
            base += ids.len() * plane.row_bytes;
        }
        self.fragments.sort_unstable_by_key(|f| (f.file, f.source));
        self.reads.clear();
        for (i, fragment) in self.fragments.iter().enumerate() {
            let block = fragment.source / BLOCK as u64 * BLOCK as u64;
            if let Some(read) = self.reads.last_mut().filter(|r| {
                r.file == fragment.file
                    && block <= r.offset + r.nbytes as u64
                    && block + BLOCK as u64 <= r.offset + CHUNK as u64
            }) {
                read.nbytes = (block - read.offset) as usize + BLOCK;
                read.required = read
                    .required
                    .max((fragment.source - read.offset) as usize + fragment.nbytes);
                read.fragments.end = i + 1;
            } else {
                self.reads.push(Read {
                    file: fragment.file,
                    offset: block,
                    nbytes: BLOCK,
                    required: (fragment.source - block) as usize + fragment.nbytes,
                    fragments: i..i + 1,
                });
            }
        }
        timing.plan = begin.elapsed().as_nanos() as u64;
        #[cfg(unix)]
        for batch in self.reads.chunks(DEPTH) {
            self.settled = false;
            let begin = Instant::now();
            for (slot, read) in batch.iter().enumerate() {
                let entry = opcode::ReadFixed::new(
                    types::Fd(self.files[read.file].as_raw_fd()),
                    self.staging.at(slot),
                    read.nbytes as u32,
                    slot as u16,
                )
                .offset(read.offset)
                .build()
                .user_data(slot as u64);
                unsafe { self.ring.submission().push(&entry) }
                    .map_err(|_| "Engram submission queue full")?;
            }
            self.stats
                .disk_reads
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            self.stats.disk_bytes.fetch_add(
                batch.iter().map(|read| read.nbytes as u64).sum(),
                Ordering::Relaxed,
            );
            timing.submit += begin.elapsed().as_nanos() as u64;
            let mut completed = 0;
            let mut failure = None;
            while completed < batch.len() {
                let begin = Instant::now();
                let result = self.ring.submit_and_wait(1);
                timing.wait += begin.elapsed().as_nanos() as u64;
                match result {
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(format!("submit Engram reads: {error}").into()),
                }
                let begin = Instant::now();
                for entry in self.ring.completion() {
                    completed += 1;
                    let slot = entry.user_data() as usize;
                    let read = &batch[slot];
                    let count = entry.result();
                    if count < 0 || (count as usize) < read.required {
                        failure = Some(format!(
                            "Engram read at {} returned {count}, requires {} bytes",
                            read.offset, read.required
                        ));
                        continue;
                    }
                    for fragment in &self.fragments[read.fragments.clone()] {
                        let source = unsafe {
                            std::slice::from_raw_parts(
                                self.staging
                                    .at(slot)
                                    .add((fragment.source - read.offset) as usize),
                                fragment.nbytes,
                            )
                        };
                        output[fragment.destination..fragment.destination + fragment.nbytes]
                            .copy_from_slice(source);
                    }
                }
                timing.complete += begin.elapsed().as_nanos() as u64;
            }
            self.settled = true;
            if let Some(error) = failure {
                return Err(error.into());
            }
        }
        #[cfg(windows)]
        {
            self.settled = false;
            self.windows
                .gather(&self.reads, &self.fragments, output, &self.stats)?;
            self.settled = true;
        }

        Ok(())
    }
}

/// One row held by [`ReadAhead`].
#[derive(Clone, Copy)]
struct HeldRow {
    id: i64,
    /// Slot of `ReadAhead::rows` holding the row.
    slot: u32,
    /// Lowest part of the announcement being applied that lists the row, or `UNLISTED`.
    part: u32,
}

/// The part of a held row that the announcement being applied does not list.
const UNLISTED: u32 = u32::MAX;

/// Rows read from disk before the gathers predicted to request them, until a gather takes
/// them. A row stays held while successive announcements keep listing it, so an
/// announcement that advances by less than its extent, or that is made for another
/// prompt, reads only the rows it adds.
struct ReadAhead {
    /// Rows held, ascending by ID between announcements.
    held: Vec<HeldRow>,
    /// Slots of `rows` holding no row.
    free: Vec<u32>,
    /// `capacity` slots, each one weight row followed by its scale row.
    rows: Vec<u8>,
    /// Rows of the announcement being applied that neither the read-ahead nor the row
    /// cache holds, each with the first part that lists it. At most `listing` entries.
    wanted: Vec<(i64, u32)>,
    /// IDs of the segment being read.
    segment: Vec<i64>,
    /// Most rows held at once.
    capacity: usize,
    /// Most rows that one announcement can add before duplicates are removed.
    listing: usize,
}

/// Copy one row to every position of `output` that requested it, then offer it to the cache.
/// `output` holds one weight row per requested position, followed by the scale rows.
fn deliver(
    cache: &mut RowCache,
    stats: &Stats,
    output: &mut [u8],
    positions: &[(i64, usize)],
    id: i64,
    weights: &[u8],
    scales: &[u8],
) {
    let scale_base = output.len() / (weights.len() + scales.len()) * weights.len();
    for &(_, row) in positions {
        output[row * weights.len()..(row + 1) * weights.len()].copy_from_slice(weights);
        output[scale_base + row * scales.len()..scale_base + (row + 1) * scales.len()]
            .copy_from_slice(scales);
    }
    // Only rows from complete successful disk reads enter the immutable cache.
    let inserted = cache.insert(id, weights, scales);
    stats
        .resident_rows
        .fetch_add(u64::from(inserted.added), Ordering::Relaxed);
    stats
        .evictions
        .fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
}

struct CachedReader {
    disk: DiskReader,
    cache: RowCache,
    requests: Vec<(i64, usize)>,
    miss_ids: Vec<i64>,
    miss_groups: Vec<(usize, usize)>,
    miss_output: Vec<u8>,
    /// Present when the table's row IDs can be derived before its gathers.
    read_ahead: Option<ReadAhead>,
    /// Row-cache misses of one gather found in `read_ahead`, ascending by the position of
    /// the row in its `held` list, each with the range of `requests` asking for that row.
    ahead_groups: Vec<(usize, usize, usize)>,
    max_rows: usize,
}

impl CachedReader {
    fn new(
        disk: DiskReader,
        planes: &[Plane; 2],
        budget: usize,
        max_rows: usize,
        read_ahead_rows: usize,
    ) -> Result<Self> {
        if max_rows == 0 {
            return Err("embedding row cache requires a nonzero gather capacity".into());
        }
        let row_bytes = planes.iter().map(|plane| plane.row_bytes).sum::<usize>();
        let mut workspace = max_rows
            .checked_mul(
                size_of::<(i64, usize)>()
                    + size_of::<i64>()
                    + size_of::<(usize, usize)>()
                    + row_bytes,
            )
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("embedding row cache workspace size overflow")?;
        // Read-ahead holds up to `read_ahead_rows` rows beside the miss workspace, within a
        // quarter of the bytes left for cached rows: a small budget keeps most of its
        // cache and reads ahead fewer rows, or none. Its listing scratch covers the whole
        // request, so that rows repeated within one step do not use up a smaller capacity.
        let segment_rows = disk.batch_rows().min(max_rows);
        if u32::try_from(read_ahead_rows).is_err() {
            return Err("embedding row read-ahead lists at most 2^32 - 1 rows of one step".into());
        }
        let listing_bytes = read_ahead_rows
            .checked_mul(size_of::<(i64, u32)>())
            .and_then(|n| n.checked_add(segment_rows * size_of::<i64>()))
            .ok_or("embedding row cache workspace size overflow")?;
        let held_bytes = size_of::<HeldRow>()
            + size_of::<u32>()
            + size_of::<(usize, usize, usize)>()
            + row_bytes;
        let capacity = budget
            .checked_sub(workspace)
            .and_then(|payload| (payload / 4).checked_sub(listing_bytes))
            .map_or(0, |bytes| (bytes / held_bytes).min(read_ahead_rows));
        if capacity != 0 {
            workspace += listing_bytes + capacity * held_bytes;
        }
        let payload_budget = budget
            .checked_sub(workspace)
            .ok_or("embedding row cache budget cannot hold its bounded miss workspace")?;
        fn allocated<T: Copy>(len: usize, value: T) -> Result<Vec<T>> {
            let mut data = Vec::new();
            data.try_reserve_exact(len)
                .map_err(|e| format!("embedding row cache workspace: {e}"))?;
            data.resize(len, value);
            // Touch the entire workspace before KV sizing, including vectors
            // whose length is cleared after allocation.
            for index in (0..len).step_by((4096 / size_of::<T>()).max(1)) {
                unsafe { data.as_mut_ptr().add(index).write_volatile(data[index]) };
            }
            Ok(data)
        }
        let mut requests = allocated(max_rows, (0i64, 0usize))?;
        let mut miss_ids = allocated(max_rows, 0i64)?;
        let mut miss_groups = allocated(max_rows, (0usize, 0usize))?;
        requests.clear();
        miss_ids.clear();
        miss_groups.clear();
        let miss_output = allocated(max_rows * row_bytes, 0u8)?;
        let mut ahead_groups = Vec::new();
        let read_ahead = if capacity != 0 {
            let unheld = HeldRow {
                id: 0,
                slot: 0,
                part: UNLISTED,
            };
            let mut held = allocated(capacity, unheld)?;
            let mut free = allocated(capacity, 0u32)?;
            let mut wanted = allocated(read_ahead_rows, (0i64, 0u32))?;
            let mut segment = allocated(segment_rows, 0i64)?;
            ahead_groups = allocated(capacity, (0usize, 0usize, 0usize))?;
            held.clear();
            free.clear();
            // Slots are taken from the end of the list, lowest slot first.
            free.extend((0..capacity as u32).rev());
            wanted.clear();
            segment.clear();
            ahead_groups.clear();
            Some(ReadAhead {
                held,
                free,
                rows: allocated(capacity * row_bytes, 0u8)?,
                wanted,
                segment,
                capacity,
                listing: read_ahead_rows,
            })
        } else {
            None
        };
        let cache = RowCache::new(
            payload_budget,
            planes[0].nbytes / planes[0].row_bytes,
            row_bytes,
        )?;
        let bytes = cache.bytes() + workspace;
        disk.stats
            .capacity_rows
            .fetch_add(cache.capacity() as u64, Ordering::Relaxed);
        disk.stats.prefetch_capacity_rows.fetch_add(
            read_ahead.as_ref().map_or(0, |ahead| ahead.capacity as u64),
            Ordering::Relaxed,
        );
        disk.stats
            .cache_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(Self {
            disk,
            cache,
            requests,
            miss_ids,
            miss_groups,
            miss_output,
            read_ahead,
            ahead_groups,
            max_rows,
        })
    }

    /// Apply an announcement of the rows that following gathers will request. `fill`
    /// reports every announced row with the ordinal of its part, a group of rows needed
    /// together and sooner than the rows of higher parts, in ascending order of parts; it
    /// stops once the sink returns false. Rows outside the table are ignored: the list
    /// is a prediction, and a gather alone validates what its graph requests.
    ///
    /// A held row stays held if the announcement lists it and is dropped otherwise. The
    /// listed rows that neither the read-ahead nor the row cache holds are then read from
    /// disk, lowest part first and in ascending row ID within a part, in segments of
    /// rows of one part whose reads normally fit one batch. The rows of a segment that
    /// has been read take the free slots. Rows of the first part, the rows wanted
    /// soonest, then take the slots of held rows that only higher parts list, highest
    /// part first: a held row gives up its slot when the row replacing it has been read.
    /// Rows of every other part take free slots only, so that the rows held for one
    /// part are never dropped for rows that another part wants no sooner than the
    /// following gather. Rows that would find no slot are not read.
    ///
    /// `interrupted` is asked while the announcement is listed, between the steps that
    /// follow, and before each segment. Once it reports true the pass ends: an incomplete
    /// listing changes nothing, and the segments already read stay held. A segment that
    /// has started is read whole, so a gather waits for its reads and none is repeated.
    /// A read that fails after all its I/O completed ends the pass the same way, because
    /// the gather that needs those rows reads them again and reports its own failure;
    /// only a failure that leaves reads outstanding is returned. The row cache and its
    /// usage counts do not change.
    fn read_ahead(
        &mut self,
        planes: &[Plane; 2],
        fill: impl FnOnce(&mut dyn FnMut(u32, i64) -> bool),
        interrupted: &dyn Fn() -> bool,
    ) -> Result<()> {
        let Some(ahead) = &mut self.read_ahead else {
            return Ok(());
        };
        let ReadAhead {
            held,
            free,
            rows,
            wanted,
            segment,
            capacity,
            listing,
        } = ahead;
        let begin = Instant::now();
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let row_bytes = weights + scales;
        let table_rows = (planes[0].nbytes / weights) as i64;
        wanted.clear();
        for row in held.iter_mut() {
            row.part = UNLISTED;
        }
        let mut reported = 0usize;
        // Held rows that no part lists yet, and the highest part that lists one.
        let mut unlisted = held.len();
        let mut last_part = 0;
        let mut complete = true;
        let cache = &self.cache;
        fill(&mut |part: u32, id: i64| {
            reported += 1;
            if reported % 64 == 0 && interrupted() {
                complete = false;
                return false;
            }
            if (0..table_rows).contains(&id) {
                match held.binary_search_by_key(&id, |row| row.id) {
                    Ok(position) if held[position].part == UNLISTED => {
                        held[position].part = part;
                        unlisted -= 1;
                        last_part = last_part.max(part);
                    }
                    Ok(_) => {}
                    // A cached row needs no slot and takes no room in the listing, so a
                    // window whose rows are cached leaves the slots to later windows.
                    Err(_) if wanted.len() < *listing && !cache.contains(id) => {
                        wanted.push((id, part))
                    }
                    Err(_) => {}
                }
            }
            // Once every held row is listed and no further row can be recorded or take a
            // slot, the rest of the announcement changes nothing and is not derived.
            unlisted != 0
                || part < last_part
                || (wanted.len() < *listing && !free.is_empty())
        });
        let mut dropped = 0;
        let mut read = 0;
        let mut result = Ok(complete && !interrupted());
        if matches!(result, Ok(true)) {
            held.retain(|row| {
                if row.part == UNLISTED {
                    free.push(row.slot);
                    dropped += 1;
                }
                row.part != UNLISTED
            });
            // Ascending by ID, keeping the lowest part that lists each row.
            wanted.sort_unstable();
            wanted.dedup_by_key(|entry| entry.0);
            result = Ok(!interrupted());
        }
        if matches!(result, Ok(true)) {
            wanted.sort_unstable_by_key(|&(id, part)| (part, id));
            result = Ok(!interrupted());
        }
        // Held rows that may still give up their slots: `held[..replaceable]`, ascending
        // by part. The rows read by this pass follow them.
        let mut replaceable = held.len();
        if matches!(result, Ok(true)) {
            // The rows of the first part that the free slots cannot hold take the slots
            // of the rows that only higher parts list. The wanted row at
            // `free.len() + n` replaces the held row with the n-th highest part.
            held.sort_unstable_by_key(|row| row.part);
            let mut slots = free.len();
            while slots < wanted.len()
                && wanted[slots].1 == 0
                && slots - free.len() < held.len()
                && held[held.len() - 1 - (slots - free.len())].part > 0
            {
                slots += 1;
            }
            wanted.truncate(slots);
        } else {
            wanted.clear();
        }
        let segment_rows = self.disk.batch_rows().min(self.max_rows);
        while read < wanted.len() && matches!(result, Ok(true)) {
            if interrupted() {
                result = Ok(false);
                break;
            }
            // A segment holds rows of one part, so a part is complete before the next starts.
            let part = wanted[read].1;
            let count = wanted[read..]
                .iter()
                .take(segment_rows)
                .take_while(|entry| entry.1 == part)
                .count();
            segment.clear();
            segment.extend(wanted[read..read + count].iter().map(|entry| entry.0));
            // The miss workspace is idle between gathers and holds one segment.
            let staged = &mut self.miss_output[..count * row_bytes];
            result = self
                .disk
                .gather(planes, &segment[..], staged)
                .map(|()| true);
            if matches!(result, Ok(true)) {
                for (index, &id) in segment.iter().enumerate() {
                    let slot = match free.pop() {
                        Some(slot) => slot,
                        // `wanted` was cut to the free slots and the rows it may replace.
                        None => {
                            replaceable -= 1;
                            dropped += 1;
                            held.swap_remove(replaceable).slot
                        }
                    };
                    let row = &mut rows[slot as usize * row_bytes..][..row_bytes];
                    row[..weights].copy_from_slice(&staged[index * weights..][..weights]);
                    row[weights..]
                        .copy_from_slice(&staged[count * weights + index * scales..][..scales]);
                    held.push(HeldRow { id, slot, part });
                }
                read += count;
            }
        }
        // A failure with reads outstanding leaves the reader unusable. Any other failure
        // loses only the rows of its segment.
        let unusable = result.is_err() && !self.disk.settled;
        if unusable {
            held.clear();
            free.clear();
            free.extend((0..*capacity as u32).rev());
            read = 0;
        }
        held.sort_unstable_by_key(|row| row.id);
        let stats = &self.disk.stats;
        stats.prefetch_rows.fetch_add(read as u64, Ordering::Relaxed);
        stats.prefetch_unused.fetch_add(dropped, Ordering::Relaxed);
        stats
            .prefetch_interrupted
            .fetch_add(u64::from(matches!(result, Ok(false))), Ordering::Relaxed);
        stats
            .prefetch_errors
            .fetch_add(u64::from(result.is_err()), Ordering::Relaxed);
        stats.prefetch_ns.fetch_add(
            begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        match result {
            Err(error) if unusable => Err(error),
            _ => Ok(()),
        }
    }

    fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if ids.len() > self.max_rows {
            return Err("embedding gather exceeds the row cache workspace capacity".into());
        }
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let scale_base = ids.len() * weights;
        self.requests.clear();
        self.miss_ids.clear();
        self.miss_groups.clear();
        self.ahead_groups.clear();
        let held: &[HeldRow] = match &self.read_ahead {
            Some(ahead) => &ahead.held,
            None => &[],
        };
        let mut cursor = 0;
        for (row, &id) in ids.iter().enumerate() {
            if id < 0 {
                output[row * weights..(row + 1) * weights].fill(0);
                output[scale_base + row * scales..scale_base + (row + 1) * scales].fill(0);
            } else {
                self.requests.push((id, row));
            }
        }
        self.requests.sort_unstable_by_key(|&(id, _)| id);
        let mut begin = 0;
        let mut hits = 0;
        while begin < self.requests.len() {
            let id = self.requests[begin].0;
            let end = begin + self.requests[begin..].partition_point(|&(next, _)| next == id);
            if let Some(payload) = self.cache.get(id) {
                hits += 1;
                for &(_, row) in &self.requests[begin..end] {
                    output[row * weights..(row + 1) * weights].copy_from_slice(&payload[..weights]);
                    output[scale_base + row * scales..scale_base + (row + 1) * scales]
                        .copy_from_slice(&payload[weights..]);
                }
            } else {
                // Misses and rows read ahead both ascend, so the search resumes at the cursor.
                cursor += held[cursor..].partition_point(|row| row.id < id);
                if held.get(cursor).is_some_and(|row| row.id == id) {
                    self.ahead_groups.push((cursor, begin, end));
                } else {
                    self.miss_ids.push(id);
                    self.miss_groups.push((begin, end));
                }
            }
            begin = end;
        }
        let stats = &self.disk.stats;
        stats.hits.fetch_add(hits, Ordering::Relaxed);
        stats.misses.fetch_add(
            (self.miss_ids.len() + self.ahead_groups.len()) as u64,
            Ordering::Relaxed,
        );
        stats
            .prefetch_hits
            .fetch_add(self.ahead_groups.len() as u64, Ordering::Relaxed);
        // Cache lookups precede every insertion, so a row of this gather is never evicted
        // before the gather reads it.
        let count = self.miss_ids.len();
        if count != 0 {
            self.disk.gather(
                planes,
                &self.miss_ids,
                &mut self.miss_output[..count * (weights + scales)],
            )?;
        }
        for (index, (&id, &(begin, end))) in self.miss_ids.iter().zip(&self.miss_groups).enumerate()
        {
            deliver(
                &mut self.cache,
                &self.disk.stats,
                output,
                &self.requests[begin..end],
                id,
                &self.miss_output[index * weights..(index + 1) * weights],
                &self.miss_output
                    [count * weights + index * scales..count * weights + (index + 1) * scales],
            );
        }
        if let Some(ahead) = &mut self.read_ahead {
            for &(position, begin, end) in &self.ahead_groups {
                let HeldRow { id, slot, .. } = ahead.held[position];
                let (weight_row, scale_row) = ahead.rows
                    [slot as usize * (weights + scales)..][..weights + scales]
                    .split_at(weights);
                deliver(
                    &mut self.cache,
                    &self.disk.stats,
                    output,
                    &self.requests[begin..end],
                    id,
                    weight_row,
                    scale_row,
                );
                // The row is served and offered to the cache: read-ahead keeps only rows
                // that no gather has taken, and the slot is free for those.
                ahead.free.push(slot);
            }
            // The positions taken ascend, as the misses that found them do.
            let mut taken = self.ahead_groups.iter().map(|group| group.0).peekable();
            let mut position = 0;
            ahead.held.retain(|_| {
                let keep = taken.peek() != Some(&position);
                if !keep {
                    taken.next();
                }
                position += 1;
                keep
            });
        }
        Ok(())
    }
}

enum Data {
    Disk(DiskReader),
    Ram([Arc<Resident>; 2]),
    Cached(Box<CachedReader>),
}

fn gather_resident(data: &Resident, row_bytes: usize, ids: &[i64], output: &mut [u8]) {
    let data = data.bytes();
    for (row, &id) in ids.iter().enumerate() {
        let target = &mut output[row * row_bytes..(row + 1) * row_bytes];
        if id < 0 {
            target.fill(0);
        } else {
            let start = id as usize * row_bytes;
            target.copy_from_slice(&data[start..start + row_bytes]);
        }
    }
}
pub(crate) struct Reader {
    data: Data,
    stats: Arc<Stats>,
}
impl Reader {
    #[cfg(test)]
    fn open(
        path: &Path,
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        Self::from_files(
            &[Arc::new(open_file(path)?)],
            planes,
            resident,
            cache_bytes,
            max_rows,
            stats,
            1,
            false,
            0,
        )
    }
    /// A nonzero `read_ahead_rows` reserves, inside `cache_bytes`, room for up to that many
    /// rows read before the gathers that request them: as many as fit a quarter of the
    /// bytes the budget leaves for cached rows. Nothing is reserved without a row cache
    /// or with the Windows readers.
    pub fn from_files(
        files: &[Arc<File>],
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
        workers: usize,
        resident_scales: bool,
        read_ahead_rows: usize,
    ) -> Result<Self> {
        if resident && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        if planes.iter().any(|plane| plane.file >= files.len()) {
            return Err("embedding plane references an unknown file".into());
        }
        let data = if resident {
            Data::Ram([
                Arc::new(Resident::load(&files[planes[0].file], planes[0])?),
                Arc::new(Resident::load(&files[planes[1].file], planes[1])?),
            ])
        } else {
            let mut disk = DiskReader::from_files(files, stats.clone(), workers)?;
            if resident_scales {
                disk.resident_scales = Some(Resident::load(&files[planes[1].file], planes[1])?);
                stats.resident_scale_bytes.fetch_add(planes[1].resident_bytes()? as u64, Ordering::Relaxed);
            }
            if cache_bytes != 0 {
                Data::Cached(Box::new(CachedReader::new(
                    disk,
                    planes,
                    cache_bytes,
                    max_rows,
                    if cfg!(unix) { read_ahead_rows } else { 0 },
                )?))
            } else {
                Data::Disk(disk)
            }
        };
        Ok(Self { data, stats })
    }
    /// Whether [`Self::read_ahead`] can hold rows for following gathers.
    pub fn reads_ahead(&self) -> bool {
        matches!(&self.data, Data::Cached(reader) if reader.read_ahead.is_some())
    }
    /// Apply an announcement of the rows that following gathers will request: keep the
    /// held rows it lists, drop the others, and read from disk the listed rows that
    /// neither the read-ahead nor the row cache holds, into the free slots and, for rows
    /// of the first part, those of held rows that only later parts list. `fill` reports
    /// each row with the ordinal of its part, a group of rows needed together, lowest
    /// part first; lower parts are read first. Rows outside the table are ignored. Once `interrupted` reports true the
    /// pass ends, keeping every segment of rows it has read. The result is an error only
    /// when a failure leaves the reader unusable. A reader that cannot hold rows does
    /// nothing and does not call `fill`.
    pub fn read_ahead(
        &mut self,
        planes: &[Plane; 2],
        fill: impl FnOnce(&mut dyn FnMut(u32, i64) -> bool),
        interrupted: &dyn Fn() -> bool,
    ) -> Result<()> {
        match &mut self.data {
            Data::Cached(reader) => reader.read_ahead(planes, fill, interrupted),
            Data::Disk(_) | Data::Ram(_) => Ok(()),
        }
    }
    /// Give up read-ahead after a pass ended outside its disk reads, by a panic in the
    /// row derivation or in the bookkeeping of held rows: forget the rows held and hold
    /// none from here on. Gathers use neither, so they continue from the row cache and
    /// the disk. Returns false and changes nothing when a failure left reads outstanding,
    /// which leaves the reader unusable for gathers too, or when it holds no rows.
    pub fn drop_read_ahead(&mut self) -> bool {
        let Data::Cached(reader) = &mut self.data else {
            return false;
        };
        if !reader.disk.settled {
            return false;
        }
        let Some(ahead) = reader.read_ahead.take() else {
            return false;
        };
        reader.ahead_groups.clear();
        let stats = &reader.disk.stats;
        stats
            .prefetch_capacity_rows
            .fetch_sub(ahead.capacity as u64, Ordering::Relaxed);
        stats
            .prefetch_unused
            .fetch_add(ahead.held.len() as u64, Ordering::Relaxed);
        stats.prefetch_errors.fetch_add(1, Ordering::Relaxed);
        true
    }
    /// Stand in for a failure that left reads in the kernel.
    #[cfg(test)]
    pub(super) fn leave_reads_outstanding(&mut self) {
        if let Data::Cached(reader) = &mut self.data {
            reader.disk.settled = false;
        }
    }
    #[cfg(unix)]
    pub(super) fn resident(data: [Arc<Resident>; 2], stats: Arc<Stats>) -> Self {
        Self {
            data: Data::Ram(data),
            stats,
        }
    }
    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if output.len() != ids.len() * planes.iter().map(|p| p.row_bytes).sum::<usize>() {
            return Err("Engram staging size differs from row count".into());
        }
        for plane in planes {
            if ids
                .iter()
                .any(|id| *id < -1 || (*id >= 0 && *id as usize >= plane.nbytes / plane.row_bytes))
            {
                return Err("Engram row exceeds its table".into());
            }
        }
        let begin = Instant::now();
        let result = match &mut self.data {
            Data::Disk(reader) => reader.gather(planes, ids, output),
            Data::Cached(reader) => reader.gather(planes, ids, output),
            Data::Ram(data) => {
                let mut base = 0;
                for (plane, data) in planes.iter().zip(data) {
                    gather_resident(data, plane.row_bytes, ids, &mut output[base..]);
                    base += ids.len() * plane.row_bytes;
                }
                Ok(())
            }
        };
        self.stats.gathers.fetch_add(1, Ordering::Relaxed);
        self.stats.gather_ns.fetch_add(
            begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    #[test]
    fn resident_scales_preserve_bytes_and_cache_hits_without_reading_the_scale_file() {
        for (weight_row_bytes, scale_row_bytes) in [(256, 8), (80, 10)] {
            check_resident_scale_rows(weight_row_bytes, scale_row_bytes);
        }
    }

    fn check_resident_scale_rows(weight_row_bytes: usize, scale_row_bytes: usize) {
        let root = std::env::temp_dir().join(format!("helios-engram-scales-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 4097;
        let planes = [
            Plane { file: 0, offset: 4093, nbytes: rows * weight_row_bytes, row_bytes: weight_row_bytes },
            Plane { file: 1, offset: 4093, nbytes: rows * scale_row_bytes, row_bytes: scale_row_bytes },
        ];
        let row_bytes = weight_row_bytes + scale_row_bytes;
        let data: Vec<Vec<u8>> = planes.iter().map(|plane| {
            (0..plane.offset as usize + plane.nbytes).map(|i| (i * 19 + i / 257) as u8).collect()
        }).collect();
        for cache in [0, 65536] {
            let mut files = Vec::new();
            for (i, bytes) in data.iter().enumerate() {
                let path = root.join(i.to_string());
                std::fs::write(&path, bytes).unwrap();
                files.push(Arc::new(open_file(&path).unwrap()));
            }
            let stats = Arc::new(Stats::default());
            let mut reader = Reader::from_files(&files, &planes, false, cache, 8, stats.clone(), 1, true, 0).unwrap();
            assert_eq!(stats.resident_scale_bytes.load(Ordering::Relaxed), planes[1].resident_bytes().unwrap() as u64);
            OpenOptions::new().write(true).open(root.join("1")).unwrap().set_len(0).unwrap();
            for ids in [vec![0, 4096, -1, 0], vec![4096, 0, -1, 0], vec![17, -1, 19]] {
                let mut expected = Vec::new();
                for (plane, bytes) in planes.iter().zip(&data) {
                    for &id in &ids {
                        if id < 0 {
                            expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                        } else {
                            let start = plane.offset as usize + id as usize * plane.row_bytes;
                            expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                        }
                    }
                }
                let mut output = vec![0xa5; expected.len()];
                reader.gather(&planes, &ids, &mut output).unwrap();
                assert_eq!(output, expected);
            }
            assert!(stats.disk_reads.load(Ordering::Relaxed) > 0);
            if cache > 0 {
                assert_eq!(stats.hits.load(Ordering::Relaxed), 2);
                assert_eq!(stats.misses.load(Ordering::Relaxed), 4);
                let reads = stats.disk_reads.load(Ordering::Relaxed);
                reader.gather(&planes, &[17, 19], &mut vec![0; row_bytes * 2]).unwrap();
                assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
            }
            assert!(reader.gather(&planes, &[rows as i64], &mut vec![0; row_bytes]).is_err());
            assert!(reader.gather(&planes, &[-2], &mut vec![0; row_bytes]).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn unsealed_resident_file() -> File {
        let descriptor = unsafe {
            libc::memfd_create(
                c"helios-resident-seal-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(descriptor >= 0);
        let file = unsafe { File::from_raw_fd(descriptor) };
        file.set_len(BLOCK as u64).unwrap();
        file
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_waits_for_unmapped_pages_held_by_io() {
        use std::io::Read;
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        let file = unsealed_resident_file();
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        unsafe { std::ptr::write_bytes(mapping.address as *mut u8, 0x39, BLOCK) };
        let mut descriptors = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let mut reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        let vector = libc::iovec {
            iov_base: mapping.address as *mut libc::c_void,
            iov_len: BLOCK,
        };
        assert_eq!(
            unsafe { libc::vmsplice(writer.as_raw_fd(), &vector, 1, 0) },
            BLOCK as isize
        );
        drop(mapping);
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EBUSY)
        );

        std::thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let file = &file;
            scope.spawn(move || send.send(seal_resident(file)).unwrap());
            // The I/O reference remains live beyond the kernel's initial pin wait.
            assert!(matches!(
                receive.recv_timeout(Duration::from_millis(250)),
                Err(RecvTimeoutError::Timeout)
            ));
            let mut bytes = [0; BLOCK];
            reader.read_exact(&mut bytes).unwrap();
            assert_eq!(bytes, [0x39; BLOCK]);
            receive.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        });
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) } & RESIDENT_SEALS,
            RESIDENT_SEALS
        );
        assert!(file.set_len(1).is_err());
        let value = 0u8;
        assert_eq!(
            unsafe { libc::pwrite(file.as_raw_fd(), (&value as *const u8).cast(), 1, 0) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EPERM)
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_rejects_a_retained_writable_mapping() {
        let file = unsealed_resident_file();
        let initial_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        assert!(seal_resident(&file).is_err());
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) },
            initial_seals
        );
        drop(mapping);
        seal_resident(&file).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_rows_are_readable_without_first_access_page_faults() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-populated-{}.bin",
            std::process::id()
        ));
        let rows = 65536;
        let second = rows * 256;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 19 + i / 7) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: second,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let file = open_file(&path).unwrap();
        let original = planes.map(|plane| Arc::new(Resident::load(&file, plane).unwrap()));
        let shared = std::array::from_fn(|i| {
            Arc::new(
                Resident::from_shared(original[i].file.try_clone().unwrap(), planes[i]).unwrap(),
            )
        });
        std::fs::remove_file(path).unwrap();
        let ids: Vec<_> = (0..128).map(|i| ((i * 509 + 17) % rows) as i64).collect();
        let mut output = vec![0; ids.len() * 264];
        let mut expected = Vec::with_capacity(output.len());
        for plane in &planes {
            for &id in &ids {
                let first = plane.offset as usize + id as usize * plane.row_bytes;
                expected.extend_from_slice(&bytes[first..first + plane.row_bytes]);
            }
        }
        for data in [original, shared] {
            let mut reader = Reader::resident(data, Arc::default());
            // Prime the copy code and output storage without touching the selected table pages.
            reader.gather(&planes, &[0], &mut output[..264]).unwrap();
            output.fill(0xA5);
            let mut before = unsafe { std::mem::zeroed::<libc::rusage>() };
            let mut after = unsafe { std::mem::zeroed::<libc::rusage>() };
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut before) },
                0
            );
            reader.gather(&planes, &ids, &mut output).unwrap();
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut after) },
                0
            );
            assert_eq!(
                after.ru_minflt, before.ru_minflt,
                "row gathering faulted resident pages"
            );
            assert_eq!(after.ru_majflt, before.ru_majflt);
            assert_eq!(output, expected);
        }
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_plane_outlives_source_and_loader_and_rejects_mutation() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-shared-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..12345).map(|i| (i * 19 + i / 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane {
            file: 0,
            offset: 97,
            nbytes: 8192,
            row_bytes: 128,
        };
        let original = Resident::load(&open_file(&path).unwrap(), plane).unwrap();
        let peer = Resident::from_shared(original.file.try_clone().unwrap(), plane).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            original.file.metadata().unwrap().ino(),
            peer.file.metadata().unwrap().ino()
        );
        assert!(original.file.set_len(1).is_err());
        let value = 255u8;
        assert_eq!(
            unsafe { libc::pwrite(original.descriptor(), (&value as *const u8).cast(), 1, 97) },
            -1
        );
        let wrong = Plane {
            nbytes: plane.nbytes + BLOCK,
            ..plane
        };
        assert!(Resident::from_shared(original.file.try_clone().unwrap(), wrong).is_err());
        std::fs::remove_file(path).unwrap();
        drop(original);
        assert_eq!(peer.bytes(), &bytes[97..97 + 8192]);
    }

    #[test]
    fn resident_load_covers_parallel_chunks_and_rejects_truncated_planes() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-bulk-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 5 / 80 + 1;
        let first = 123;
        let second = first + rows * 80 + 37;
        let bytes: Vec<_> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let file = open_file(&path).unwrap();
        for plane in planes {
            let data = Resident::load(&file, plane).unwrap();
            let begin = plane.offset as usize;
            assert_eq!(data.bytes(), &bytes[begin..begin + plane.nbytes]);
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            Resident::load(&file, planes[0])
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn warm_rows_avoid_disk_and_failed_misses_never_enter_the_cache() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-cache-{}.bin", std::process::id()));
        let rows = 128;
        let first = 4093;
        let second = first + rows * 80 + 7;
        let bytes: Vec<u8> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let stats = Arc::new(Stats::default());
        let mut reader = Reader::open(&path, &planes, false, 65536, 32, stats.clone()).unwrap();
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 65536);
        fn check(reader: &mut Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
            let mut expected = Vec::new();
            for plane in planes {
                for &id in ids {
                    if id < 0 {
                        expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                    } else {
                        let start = plane.offset as usize + id as usize * plane.row_bytes;
                        expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                    }
                }
            }
            let mut output = vec![0xa5; ids.len() * 90];
            reader.gather(planes, ids, &mut output).unwrap();
            assert_eq!(output, expected);
        }
        check(&mut reader, &planes, &bytes, &[2, 9, 2, -1, 17]);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 3);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        check(&mut reader, &planes, &bytes, &[17, 9, -1, 2, 17]);
        assert_eq!(stats.hits.load(Ordering::Relaxed), 3);
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
        assert!(reader.gather(&planes, &[128], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[-2], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[2; 33], &mut [0; 33 * 90]).is_err());
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        check(&mut reader, &planes, &bytes, &[9, 2]);
        assert!(reader.gather(&planes, &[20], &mut [0; 90]).is_err());
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        std::fs::write(&path, &bytes).unwrap();
        check(&mut reader, &planes, &bytes, &[20]);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 4);
        std::fs::remove_file(path).unwrap();
    }

    /// A file holding `rows` 256-byte weight rows and their 8-byte scale rows in unaligned planes.
    fn read_ahead_table(name: &str, rows: usize) -> (std::path::PathBuf, Vec<u8>, [Plane; 2]) {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-{name}-{}.bin",
            std::process::id()
        ));
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<u8> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        (path, bytes, planes)
    }

    /// Gather `ids` and compare every delivered byte with the table file.
    fn check_gather(reader: &mut Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
        let mut expected = Vec::new();
        for plane in planes {
            for &id in ids {
                if id < 0 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let start = plane.offset as usize + id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                }
            }
        }
        let mut output = vec![0xa5; expected.len()];
        reader.gather(planes, ids, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    /// Announce `parts`, each a list of rows needed together, lowest part first.
    fn announce(
        reader: &mut Reader,
        planes: &[Plane; 2],
        parts: &[&[i64]],
        interrupted: &dyn Fn() -> bool,
    ) -> Result<()> {
        reader.read_ahead(
            planes,
            |sink| {
                for (part, ids) in parts.iter().enumerate() {
                    for &id in *ids {
                        if !sink(part as u32, id) {
                            return;
                        }
                    }
                }
            },
            interrupted,
        )
    }

    #[test]
    #[cfg(unix)]
    fn rows_read_ahead_serve_gathers_and_stay_while_predicted() {
        let _one_reader = READ_AHEAD_READER.lock().unwrap_or_else(|error| error.into_inner());
        let (path, bytes, planes) = read_ahead_table("ahead", 4097);
        let stats = Arc::new(Stats::default());
        let files = [Arc::new(open_file(&path).unwrap())];
        let mut reader =
            Reader::from_files(&files, &planes, false, 1 << 20, 64, stats.clone(), 1, false, 48)
                .unwrap();
        assert!(reader.reads_ahead());
        assert_eq!(stats.prefetch_capacity_rows.load(Ordering::Relaxed), 48);
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 1 << 20);
        let disk_reads = || stats.disk_reads.load(Ordering::Relaxed);
        let cache = || {
            (
                stats.hits.load(Ordering::Relaxed),
                stats.misses.load(Ordering::Relaxed),
                stats.resident_rows.load(Ordering::Relaxed),
            )
        };
        // Rows read ahead, cache misses served from them, and rows dropped unused.
        let ahead = || {
            (
                stats.prefetch_rows.load(Ordering::Relaxed),
                stats.prefetch_hits.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
            )
        };

        check_gather(&mut reader, &planes, &bytes, &[5, 9]);
        assert_eq!(cache(), (0, 2, 2));
        let gathered = disk_reads();

        // The prediction repeats a row and names cached rows and rows outside the table.
        // Rows 17, 40, 300, 500 and 4096 are read; the row cache and its counters do not
        // change.
        let predicted = [40, 9, 4096, 17, 40, -1, 4097, 5, 300, 500, i64::MAX];
        announce(&mut reader, &planes, &[&predicted], &|| false).unwrap();
        assert_eq!(ahead(), (5, 0, 0));
        assert_eq!(cache(), (0, 2, 2));
        let read_ahead = disk_reads();
        assert!(read_ahead > gathered);
        assert!(stats.prefetch_ns.load(Ordering::Relaxed) > 0);

        // Three rows come from the read-ahead, row 9 from the cache, and only row 2000
        // from disk: at most two weight blocks and one scale block.
        check_gather(&mut reader, &planes, &bytes, &[40, 17, -1, 2000, 9, 4096, 40]);
        assert_eq!(ahead(), (5, 3, 0));
        assert_eq!(cache(), (1, 6, 6));
        assert!((1..=3).contains(&(disk_reads() - read_ahead)));

        // Rows served from the read-ahead entered the cache like rows read by the gather,
        // and the rows not yet requested stay available to later gathers.
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &[17, 40, 4096, 2000]);
        assert_eq!(cache(), (5, 6, 6));
        check_gather(&mut reader, &planes, &bytes, &[500, 9]);
        assert_eq!(ahead(), (5, 4, 0));
        assert_eq!(cache(), (6, 7, 7));
        assert_eq!(disk_reads(), reads);

        // A prediction that still lists row 300 keeps it and reads only row 301. Row 17
        // is cached, and the rows no longer listed had all been used.
        announce(&mut reader, &planes, &[&[300, 301, 17]], &|| false).unwrap();
        assert_eq!(ahead(), (6, 4, 0));
        assert!((1..=3).contains(&(disk_reads() - reads)));
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &[301, 17, 300]);
        assert_eq!(ahead(), (6, 6, 0));
        assert_eq!(cache(), (7, 9, 9));
        assert_eq!(disk_reads(), reads);

        // A row read ahead and dropped before any gather takes it counts as unused, and a
        // later gather reads it from disk.
        announce(&mut reader, &planes, &[&[777]], &|| false).unwrap();
        assert_eq!(ahead(), (7, 6, 0));
        announce(&mut reader, &planes, &[&[888]], &|| false).unwrap();
        assert_eq!(ahead(), (8, 6, 1));
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &[777]);
        assert_eq!(ahead(), (8, 6, 1));
        assert!(disk_reads() > reads);

        // Free slots go to the lowest part: with row 888 held, 47 slots remain for a part
        // of 60 rows listed before a part of 10, and the 47 lowest of its IDs are read.
        let first: Vec<i64> = (1100..1160).collect();
        let second: Vec<i64> = (1000..1010).collect();
        announce(&mut reader, &planes, &[&[888], &first, &second], &|| false).unwrap();
        assert_eq!(ahead(), (8 + 47, 6, 1));
        // A gather takes rows out of the read-ahead: their slots are free again, and the
        // row cache serves them from then on.
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &first[..47]);
        assert_eq!(ahead(), (8 + 47, 6 + 47, 1));
        check_gather(&mut reader, &planes, &bytes, &first[..47]);
        assert_eq!(ahead(), (8 + 47, 6 + 47, 1));
        assert_eq!(disk_reads(), reads);

        // With the read-ahead full, only the rows of the first part take the slots of
        // rows that a later part lists. Ten rows listed in a middle part are not read,
        // and nothing is dropped.
        let later: Vec<i64> = (1200..1247).collect();
        let sooner: Vec<i64> = (1300..1310).collect();
        announce(&mut reader, &planes, &[&[888], &later], &|| false).unwrap();
        assert_eq!(ahead(), (55 + 47, 53, 1));
        let reads = disk_reads();
        announce(&mut reader, &planes, &[&[888], &sooner, &later], &|| false).unwrap();
        assert_eq!((ahead(), disk_reads()), ((102, 53, 1), reads));
        // Listed first, they are read: ten rows of the last part are dropped unused, and
        // row 888 of the part between stays.
        announce(&mut reader, &planes, &[&sooner, &[888], &later], &|| false).unwrap();
        assert_eq!(ahead(), (102 + 10, 53, 1 + 10));
        // A held row never makes room for a row of its own part: nothing is read, and
        // nothing is dropped.
        let reads = disk_reads();
        let same: Vec<i64> = [888].into_iter().chain(1200..1247).chain(1300..1310).chain(1400..1405).collect();
        announce(&mut reader, &planes, &[&same], &|| false).unwrap();
        assert_eq!(ahead(), (112, 53, 11));
        assert_eq!(disk_reads(), reads);
        check_gather(&mut reader, &planes, &bytes, &[888]);
        assert_eq!((ahead(), disk_reads()), ((112, 54, 11), reads));
        // An announcement that lists none of the 47 rows still held drops them unused.
        announce(&mut reader, &planes, &[&second], &|| false).unwrap();
        assert_eq!(ahead(), (112 + 10, 54, 11 + 47));
        // Cached rows take no room in the listing: after a first part of 48 cached rows,
        // as many as the read-ahead holds, the rows of the second part are still read.
        let cached: Vec<i64> = first[..47].iter().copied().chain([888]).collect();
        let reads = disk_reads();
        announce(&mut reader, &planes, &[&cached, &[3000, 3001]], &|| false).unwrap();
        assert_eq!(ahead(), (122 + 2, 54, 58 + 10));
        check_gather(&mut reader, &planes, &bytes, &[3001, 3000]);
        assert_eq!(ahead(), (124, 56, 68));
        assert!((1..=3).contains(&(disk_reads() - reads)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn interrupted_read_ahead_keeps_every_segment_it_read() {
        let _one_reader = READ_AHEAD_READER.lock().unwrap_or_else(|error| error.into_inner());
        // Rows of one segment: both planes of these readers are read from disk.
        const SEGMENT: usize = DEPTH / 2;
        const ROWS: u64 = SEGMENT as u64;
        let (path, bytes, planes) = read_ahead_table("yield", 16384);
        let stats = Arc::new(Stats::default());
        let files = [Arc::new(open_file(&path).unwrap())];
        let mut reader =
            Reader::from_files(&files, &planes, false, 16 << 20, 8192, stats.clone(), 1, false, 8192)
                .unwrap();
        assert_eq!(stats.prefetch_capacity_rows.load(Ordering::Relaxed), 8192);
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 16 << 20);
        let disk_reads = || stats.disk_reads.load(Ordering::Relaxed);
        // Rows read ahead, cache misses served from them, interrupted passes.
        let ahead = || {
            (
                stats.prefetch_rows.load(Ordering::Relaxed),
                stats.prefetch_hits.load(Ordering::Relaxed),
                stats.prefetch_interrupted.load(Ordering::Relaxed),
            )
        };

        // Weight rows about 16 KiB apart need one read each: a segment of rows fills
        // half a batch with weight reads and adds one read of scales. The row sets read
        // through this reader are disjoint.
        let sparse: Vec<i64> = (0..200).map(|i| i * 63).collect();
        // A gather already waiting: the prediction is not applied and no row is read.
        announce(&mut reader, &planes, &[&sparse], &|| true).unwrap();
        assert_eq!((disk_reads(), ahead()), (0, (0, 0, 1)));
        // A gather arriving during the first segment: its rows are kept, the next
        // segment is not read, and the gather takes the rows it needs from memory.
        announce(&mut reader, &planes, &[&sparse], &|| disk_reads() > 0).unwrap();
        let reads = disk_reads();
        assert!((ROWS..=DEPTH as u64).contains(&reads));
        assert_eq!(ahead(), (ROWS, 0, 2));
        check_gather(&mut reader, &planes, &bytes, &sparse[..10]);
        assert_eq!((disk_reads(), ahead()), (reads, (ROWS, 10, 2)));
        // The same prediction again reads only the 136 rows the interruption left out.
        announce(&mut reader, &planes, &[&sparse], &|| false).unwrap();
        assert_eq!(ahead(), (200, 10, 2));
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &sparse);
        assert_eq!(disk_reads(), reads);
        assert_eq!(stats.hits.load(Ordering::Relaxed) + ahead().1, 10 + 200);

        // Rows 768 bytes apart coalesce into two or three reads per segment. An
        // interruption after some segments keeps all of them: a whole number of segments,
        // lowest IDs first.
        let dense: Vec<i64> = (0..5000).map(|i| i * 3 + 1).collect();
        let before = disk_reads();
        announce(&mut reader, &planes, &[&dense], &|| disk_reads() >= before + 40).unwrap();
        let kept = (ahead().0 - 200) as usize;
        assert!(kept > 0 && kept < dense.len() && kept % SEGMENT == 0);
        assert_eq!(ahead().2, 3);
        let (reads, hits) = (disk_reads(), ahead().1);
        check_gather(&mut reader, &planes, &bytes, &dense[..kept]);
        assert_eq!((disk_reads(), ahead().1), (reads, hits + kept as u64));
        // Completing the prediction reads every remaining row once, and the gather of the
        // whole set reads none.
        announce(&mut reader, &planes, &[&dense], &|| false).unwrap();
        assert_eq!(ahead().0, 200 + 5000);
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &dense);
        assert_eq!(disk_reads(), reads);
        assert_eq!(stats.prefetch_unused.load(Ordering::Relaxed), 0);
        drop(reader);

        // A held row gives up its slot once the row replacing it has been read. With 256
        // rows of a later part filling the read-ahead, a pass interrupted after its first
        // segment of 200 rows wanted sooner has replaced one segment of rows, not 200.
        let stats = Arc::new(Stats::default());
        let mut reader =
            Reader::from_files(&files, &planes, false, 1 << 20, 256, stats.clone(), 1, false, 256)
                .unwrap();
        assert_eq!(stats.prefetch_capacity_rows.load(Ordering::Relaxed), 256);
        let disk_reads = || stats.disk_reads.load(Ordering::Relaxed);
        // Rows read ahead, cache misses served from them, rows dropped unused.
        let ahead = || {
            (
                stats.prefetch_rows.load(Ordering::Relaxed),
                stats.prefetch_hits.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
            )
        };
        let later: Vec<i64> = (0..256).map(|i| i * 63).collect();
        let sooner: Vec<i64> = (0..200).map(|i| i * 63 + 31).collect();
        announce(&mut reader, &planes, &[&later], &|| false).unwrap();
        assert_eq!(ahead(), (256, 0, 0));
        let before = disk_reads();
        announce(&mut reader, &planes, &[&sooner, &later], &|| disk_reads() > before).unwrap();
        assert_eq!(ahead(), (256 + ROWS, 0, ROWS));
        assert_eq!(stats.prefetch_interrupted.load(Ordering::Relaxed), 1);
        let reads = disk_reads();
        check_gather(&mut reader, &planes, &bytes, &sooner[..SEGMENT]);
        assert_eq!((disk_reads(), ahead()), (reads, (256 + ROWS, ROWS, ROWS)));
        // The rows that kept their slots serve their gather, which reads the others.
        check_gather(&mut reader, &planes, &bytes, &later);
        assert_eq!(ahead(), (256 + ROWS, 256, ROWS));
        assert!((ROWS..=DEPTH as u64).contains(&(disk_reads() - reads)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn read_ahead_needs_its_workspace_and_survives_failed_reads() {
        let _one_reader = READ_AHEAD_READER.lock().unwrap_or_else(|error| error.into_inner());
        let (path, bytes, planes) = read_ahead_table("ahead-errors", 128);
        let files = [Arc::new(open_file(&path).unwrap())];
        // No rows are derived or read without a row cache, without the reservation, or
        // for tables resident in RAM.
        for (resident, cache_bytes, read_ahead_rows) in
            [(false, 0, 32), (false, 1 << 20, 0), (true, 0, 32)]
        {
            let stats = Arc::new(Stats::default());
            let mut reader = Reader::from_files(
                &files,
                &planes,
                resident,
                cache_bytes,
                32,
                stats.clone(),
                1,
                false,
                read_ahead_rows,
            )
            .unwrap();
            assert!(!reader.reads_ahead());
            assert_eq!(stats.prefetch_capacity_rows.load(Ordering::Relaxed), 0);
            assert!(stats.cache_bytes.load(Ordering::Relaxed) <= cache_bytes as u64);
            reader
                .read_ahead(&planes, |_| panic!("no reader holds these rows"), &|| false)
                .unwrap();
            assert_eq!(stats.disk_reads.load(Ordering::Relaxed), 0);
        }
        // The Windows readers do not read ahead.
        if !cfg!(unix) {
            std::fs::remove_file(path).unwrap();
            return;
        }
        // A budget that cannot spare a quarter of its cache for the whole request holds
        // fewer rows, and reads ahead no more than it holds.
        let stats = Arc::new(Stats::default());
        let mut reader =
            Reader::from_files(&files, &planes, false, 65536, 64, stats.clone(), 1, false, 64)
                .unwrap();
        let capacity = stats.prefetch_capacity_rows.load(Ordering::Relaxed);
        assert!(reader.reads_ahead() && (1..64).contains(&capacity));
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 65536);
        let all: Vec<i64> = (0..64).collect();
        announce(&mut reader, &planes, &[&all], &|| false).unwrap();
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), capacity);
        drop(reader);

        let stats = Arc::new(Stats::default());
        let mut reader =
            Reader::from_files(&files, &planes, false, 1 << 20, 32, stats.clone(), 1, false, 32)
                .unwrap();
        assert!(reader.reads_ahead());
        announce(&mut reader, &planes, &[&[7, 100]], &|| false).unwrap();
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), 2);
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        // Reading row 101 fails after its I/O completed: the pass ends without it, row 7
        // stays held, and the reader remains usable.
        announce(&mut reader, &planes, &[&[7, 101]], &|| false).unwrap();
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), 2);
        assert_eq!(stats.prefetch_errors.load(Ordering::Relaxed), 1);
        assert_eq!(stats.prefetch_unused.load(Ordering::Relaxed), 1);
        // The gather that needs row 101 reads it itself, and fails while the file lacks it.
        assert!(reader.gather(&planes, &[101], &mut [0; 264]).is_err());
        std::fs::write(&path, &bytes).unwrap();
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        check_gather(&mut reader, &planes, &bytes, &[101, 7]);
        assert_eq!(stats.prefetch_hits.load(Ordering::Relaxed), 1);
        assert!(stats.disk_reads.load(Ordering::Relaxed) > reads);
        announce(&mut reader, &planes, &[&[21, 101]], &|| false).unwrap();
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), 3);
        check_gather(&mut reader, &planes, &bytes, &[21]);
        assert_eq!(stats.prefetch_hits.load(Ordering::Relaxed), 2);

        // A reader whose reads all completed gives up read-ahead and keeps serving
        // gathers: the row it held counts as unused, and later announcements read nothing.
        announce(&mut reader, &planes, &[&[33]], &|| false).unwrap();
        let (capacity, unused, errors) = (
            stats.prefetch_capacity_rows.load(Ordering::Relaxed),
            stats.prefetch_unused.load(Ordering::Relaxed),
            stats.prefetch_errors.load(Ordering::Relaxed),
        );
        assert!(capacity > 0);
        assert!(reader.drop_read_ahead());
        assert!(!reader.reads_ahead() && !reader.drop_read_ahead());
        assert_eq!(
            (
                stats.prefetch_capacity_rows.load(Ordering::Relaxed),
                stats.prefetch_unused.load(Ordering::Relaxed),
                stats.prefetch_errors.load(Ordering::Relaxed),
            ),
            (0, unused + 1, errors + 1)
        );
        let rows = stats.prefetch_rows.load(Ordering::Relaxed);
        announce(&mut reader, &planes, &[&[44]], &|| false).unwrap();
        assert_eq!(stats.prefetch_rows.load(Ordering::Relaxed), rows);
        check_gather(&mut reader, &planes, &bytes, &[33, 44, 21]);
        drop(reader);
        // With reads left outstanding the reader is unusable, and read-ahead stays.
        let stats = Arc::new(Stats::default());
        let mut reader =
            Reader::from_files(&files, &planes, false, 1 << 20, 32, stats.clone(), 1, false, 32)
                .unwrap();
        reader.leave_reads_outstanding();
        assert!(!reader.drop_read_ahead() && reader.reads_ahead());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_and_ram_gather_identical_rows_across_unaligned_planes() {
        let path = std::env::temp_dir().join(format!("helios-engram-{}.bin", std::process::id()));
        let rows = 4097;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<_> = (0..8192)
            .map(|i| {
                if i % 19 == 0 {
                    -1
                } else {
                    (i * 101 % rows) as i64
                }
            })
            .collect();
        let mut expected = Vec::new();
        for plane in &planes {
            for id in &ids {
                if *id == -1 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let offset = plane.offset as usize + *id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[offset..offset + plane.row_bytes]);
                }
            }
        }
        for (resident, cache_bytes) in [(false, 0), (true, 0), (false, 3 << 20)] {
            let mut reader = Reader::open(
                &path,
                &planes,
                resident,
                cache_bytes,
                ids.len(),
                Arc::default(),
            )
            .unwrap();
            for _ in 0..3 {
                let mut actual = vec![0xA5; expected.len()];
                reader.gather(&planes, &ids, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(
                reader
                    .gather(&planes, &[rows as i64], &mut [0; 264])
                    .is_err()
            );
        }
        let mut reader = Reader::open(&path, &planes, false, 0, ids.len(), Arc::default()).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &[100], &mut [0; 264]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
    #[test]
    fn equal_offsets_in_separate_files_never_coalesce() {
        let root = std::env::temp_dir().join(format!("helios-engram-files-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 4093,
                nbytes: 80 * 128,
                row_bytes: 80,
            },
            Plane {
                file: 1,
                offset: 4093,
                nbytes: 10 * 128,
                row_bytes: 10,
            },
        ];
        let mut files = Vec::new();
        for (i, plane) in planes.iter().enumerate() {
            let path = root.join(i.to_string());
            std::fs::write(
                &path,
                vec![i as u8 + 1; plane.offset as usize + plane.nbytes],
            )
            .unwrap();
            files.push(Arc::new(open_file(&path).unwrap()));
        }
        for (resident, cache) in [(false, 0), (false, 65536), (true, 0)] {
            let mut reader =
                Reader::from_files(&files, &planes, resident, cache, 16, Arc::default(), 1, false, 0).unwrap();
            for _ in 0..3 {
                let mut output = vec![0xa5; 4 * 90];
                reader
                    .gather(&planes, &[17, 2, -1, 17], &mut output)
                    .unwrap();
                assert_eq!(&output[..80], &[1; 80]);
                assert_eq!(&output[160..240], &[0; 80]);
                assert_eq!(&output[320..340], &[2; 20]);
                assert_eq!(&output[340..350], &[0; 10]);
                assert_eq!(&output[350..360], &[2; 10]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
