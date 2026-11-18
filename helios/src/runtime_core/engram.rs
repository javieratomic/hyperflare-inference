/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

use super::{
    format::{Memory, Core, invalid},
    protobuf::wire,
};
use crate::{
    Result,
    engram::{Head, Index, Plane, RowHash, Table},
};
use std::{collections::HashSet, sync::Arc};

/// Payload ownership follows the table layout; absent partitions retain shared storage.
pub(super) fn spans(
    index: &wire::ModelIndex,
    selected: Option<usize>,
) -> Result<Vec<(Option<usize>, &wire::Span)>> {
    let world = index
        .cluster
        .as_ref()
        .map_or(index.ranks.len(), |c| c.world as usize);
    let mut result = Vec::new();
    for table in index.engram.iter().flat_map(|e| &e.tables) {
        validate_shards(table, world)?;
        if table.shards.is_empty() {
            for span in [&table.weights, &table.scales] {
                result.push((
                    None,
                    span.as_ref()
                        .ok_or_else(|| invalid("missing table plane"))?,
                ));
            }
        } else {
            for shard in &table.shards {
                if selected.is_none_or(|rank| rank == shard.rank as usize) {
                    for span in [&shard.weights, &shard.scales] {
                        result.push((
                            Some(shard.rank as usize),
                            span.as_ref()
                                .ok_or_else(|| invalid("missing table partition plane"))?,
                        ));
                    }
                }
            }
        }
    }
    Ok(result)
}

fn validate_shards(table: &wire::EngramTable, world: usize) -> Result<()> {
    if table.shards.is_empty() {
        return Ok(());
    }
    if table.weights.is_some()
        || table.scales.is_some()
        || table.request_buffer.is_empty()
        || table.shards.len() != world
    {
        return Err(invalid("invalid partitioned embedding storage"));
    }
    let mut end = 0;
    for (rank, shard) in table.shards.iter().enumerate() {
        if shard.rank as usize != rank || shard.first_row != end || shard.rows == 0 {
            return Err(invalid(
                "embedding partitions must cover table rows in rank order",
            ));
        }
        end = end
            .checked_add(shard.rows)
            .ok_or_else(|| invalid("embedding partition overflow"))?;
        // Sibling partitions absent from a rank-selective copy carry neither plane; the
        // owning rank still requires both (decode and `spans` reject a missing plane).
        if shard.weights.is_none() != shard.scales.is_none() {
            return Err(invalid("missing embedding partition plane"));
        }
    }
    if end != table.rows {
        return Err(invalid("embedding partitions must cover all table rows"));
    }
    Ok(())
}

pub(super) fn decode<F: super::payload::Files>(
    value: wire::Engram,
    archive: &F,
    cores: &[Core],
    rank: usize,
    world: usize,
) -> Result<Index> {
    if value.compressed_vocabulary == 0
        || value
            .token_map
            .iter()
            .any(|n| *n >= value.compressed_vocabulary)
        || value.pad_token_id
            >= if value.token_map.is_empty() {
                value.compressed_vocabulary
            } else {
                value.token_map.len() as u32
            }
        || value.tables.is_empty()
    {
        return Err(invalid("invalid Engram vocabulary or table catalog"));
    }
    let mut layers = HashSet::new();
    let mut buffers = HashSet::new();
    let mut events = HashSet::new();
    let mut tables = Vec::new();
    let token_map: Arc<[u32]> = value.token_map.as_slice().into();
    for table in value.tables {
        validate_shards(&table, world)?;
        let gpu = !table.request_buffer.is_empty();
        let encoding = wire::engram_table::Encoding::try_from(table.encoding)
            .map_err(|_| invalid("unknown n-gram encoding"))?;
        let (width, block, row_bytes, scale_bytes) = match encoding {
            wire::engram_table::Encoding::Fp8E4m3E8m0 => (256, 32, 256, 8),
            wire::engram_table::Encoding::Nvfp4E4m3Global
                if table.global_scale.is_finite() && table.global_scale > 0.0 =>
            {
                (160, 16, 80, 10)
            }
            _ => return Err(invalid("invalid n-gram encoding or global scale")),
        };
        let geometry = table
            .hash
            .as_ref()
            .ok_or_else(|| invalid("n-gram table lacks hash geometry"))?;
        let order = match wire::engram_hash::Method::try_from(geometry.method) {
            Ok(wire::engram_hash::Method::XorModulo) if !value.token_map.is_empty() => 4,
            Ok(wire::engram_hash::Method::EosXorModulo) if gpu && value.token_map.is_empty() => 3,
            _ => return Err(invalid("unsupported n-gram hash method or request source")),
        };
        let heads = (order - 1) * 8;
        if !layers.insert(table.layer)
            || table.rows == 0
            || table.embedding_width != width
            || table.scale_block != block
            || !events.insert(table.ready_event.clone())
            || table.ready_event.is_empty()
            || [
                &table.weight_buffer,
                &table.scale_buffer,
                &table.hash_buffer,
            ]
            .iter()
            .any(|name| name.is_empty() || !buffers.insert((*name).clone()))
        {
            return Err(invalid("invalid Engram table encoding or bindings"));
        }
        if gpu && !buffers.insert(table.request_buffer.clone()) {
            return Err(invalid("n-gram mailbox aliases another binding"));
        }
        let geometry = table
            .hash
            .ok_or_else(|| invalid("Engram table lacks hash geometry"))?;
        if geometry.orders != (2..=order as u32).flat_map(|n| [n; 8]).collect::<Vec<_>>()
            || geometry.multipliers.len() != order
            || geometry.primes.len() != heads
            || geometry.offsets.len() != heads
            || geometry.multipliers.iter().any(|m| {
                *m == 0
                    || m.checked_mul(value.compressed_vocabulary as u64)
                        .is_none_or(|n| n > i64::MAX as u64)
            })
            || geometry.primes.iter().zip(&geometry.offsets).any(|(p, o)| {
                *p < 2
                    || o.checked_add(*p)
                        .is_none_or(|n| n > table.rows || n > i64::MAX as u64)
            })
        {
            return Err(invalid("invalid Engram hash geometry"));
        }
        let (first_row, stored_rows, weights, scales) = if table.shards.is_empty() {
            (0, table.rows, table.weights.clone(), table.scales.clone())
        } else {
            let shard = table
                .shards
                .get(rank)
                .ok_or_else(|| invalid("embedding partition rank is absent"))?;
            (shard.first_row, shard.rows, shard.weights.clone(), shard.scales.clone())
        };
        // XOR-modulo rows depend only on prompt tokens, so the host can derive them before
        // a graph publishes its requests. The orders, multipliers, primes and offsets were
        // validated above; EOS-bounded tables keep graph-published requests only.
        let hash = (geometry.method == wire::engram_hash::Method::XorModulo as i32).then(|| {
            RowHash {
                token_map: token_map.clone(),
                pad: value.pad_token_id,
                multipliers: geometry.multipliers.clone(),
                heads: local_heads(&geometry, first_row, stored_rows),
            }
        });
        let plane = |span: Option<wire::Span>, row_bytes: usize| -> Result<Plane> {
            let span = span.ok_or_else(|| invalid("Engram table lacks a payload plane"))?;
            if stored_rows.checked_mul(row_bytes as u64) != Some(span.nbytes) {
                return Err(invalid("Engram payload geometry mismatch"));
            }
            Ok(Plane {
                file: archive.source_index(&span)?,
                offset: archive.source(&span)?.1,
                nbytes: usize::try_from(span.nbytes)
                    .map_err(|_| invalid("Engram plane exceeds host range"))?,
                row_bytes,
            })
        };
        let planes = [plane(weights, row_bytes)?, plane(scales, scale_bytes)?];
        for core in cores {
            if !core.events.contains(&table.ready_event) {
                return Err(invalid("rank lacks Engram readiness event"));
            }
            let execution = core
                .execution
                .as_ref()
                .ok_or_else(|| invalid("Engram rank lacks execution bindings"))?;
            let tokens = execution
                .graphs
                .iter()
                .filter(|g| {
                    matches!(
                        g.kind.as_str(),
                        "prefill" | "decode" | "speculative" | "mixed"
                    )
                })
                .map(|g| g.packed_tokens())
                .max()
                .ok_or_else(|| invalid("Engram catalog lacks consumers"))?;
            for (name, dtype, columns) in [
                (&table.weight_buffer, "uint8", row_bytes as u64),
                (&table.scale_buffer, "uint8", scale_bytes as u64),
                (&table.hash_buffer, "int64", heads as u64),
            ] {
                let buffer = execution
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or_else(|| invalid("rank lacks Engram input buffer"))?;
                let rows = if dtype == "uint8" {
                    tokens * heads as u64
                } else {
                    tokens
                };
                if buffer.dtype != dtype
                    || buffer.shape.len() != 2
                    || buffer.shape[0] < rows
                    || buffer.shape[1] != columns
                    || buffer.strides != [columns, 1]
                    || buffer.nbytes
                        != buffer.shape[0]
                            .checked_mul(columns * if dtype == "int64" { 8 } else { 1 })
                            .unwrap_or(0)
                {
                    return Err(invalid("invalid compact Engram buffer layout"));
                }
            }
            if gpu {
                let request = execution
                    .buffers
                    .iter()
                    .find(|b| b.name == table.request_buffer)
                    .ok_or_else(|| invalid("rank lacks n-gram request mailbox"))?;
                let reference = request
                    .reference
                    .as_ref()
                    .ok_or_else(|| invalid("mailbox lacks storage"))?;
                if request.dtype != "int64"
                    || request.shape.len() != 1
                    || request.strides != [1]
                    || request.shape[0] < tokens * heads as u64 + 1
                    || request.nbytes != request.shape[0] * 8
                    || !reference.offset.is_multiple_of(8)
                    || !core.allocations.iter().any(|a| {
                        a.name == reference.allocation && matches!(a.memory, Memory::MappedHost)
                    })
                {
                    return Err(invalid("invalid mapped n-gram mailbox"));
                }
            }
            for graph in execution.graphs.iter().filter(|g| {
                matches!(
                    g.kind.as_str(),
                    "prefill" | "decode" | "speculative" | "mixed"
                )
            }) {
                if !gpu
                    && !graph.input_bindings.iter().any(|b| {
                        b.buffer_name == table.hash_buffer
                            && b.nbytes >= graph.packed_tokens() * heads as u64 * 8
                    })
                {
                    return Err(invalid("Engram graph lacks hash input binding"));
                }
            }
        }
        tables.push(Table {
            planes,
            heads: geometry.primes.len(),
            weights: table.weight_buffer,
            scales: table.scale_buffer,
            hashes: table.hash_buffer,
            event: table.ready_event,
            request: gpu.then_some(table.request_buffer),
            hash,
        });
    }
    Ok(Index { tables })
}

/// Lookup heads with the partition-local first row of each head stored in the rows
/// `first_row..first_row + stored_rows` of the table. Published offsets index the table's
/// partitions concatenated in rank order, and row requests are partition-local.
fn local_heads(geometry: &wire::EngramHash, first_row: u64, stored_rows: u64) -> Vec<Head> {
    geometry
        .orders
        .iter()
        .zip(&geometry.primes)
        .zip(&geometry.offsets)
        .map(|((&order, &prime), &offset)| Head {
            order: order as usize,
            prime,
            first_row: offset.checked_sub(first_row).filter(|local| {
                local
                    .checked_add(prime)
                    .is_some_and(|end| end <= stored_rows)
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn partitioned() -> wire::EngramTable {
        wire::EngramTable {
            rows: 40,
            request_buffer: "requests".into(),
            shards: (0..4)
                .map(|rank| wire::EngramTableShard {
                    rank,
                    first_row: rank as u64 * 10,
                    rows: 10,
                    weights: Some(wire::Span {
                        nbytes: 2560,
                        ..Default::default()
                    }),
                    scales: Some(wire::Span {
                        nbytes: 80,
                        ..Default::default()
                    }),
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn table_partitions_cover_rows_exactly_once() {
        let table = partitioned();
        assert!(validate_shards(&table, 4).is_ok());
        assert!(validate_shards(&table, 2).is_err());
        for change in 0..7 {
            let mut invalid = table.clone();
            match change {
                0 => invalid.shards[1].first_row += 1,
                1 => invalid.shards[1].first_row -= 1,
                2 => invalid.shards[2].rank = 0,
                3 => invalid.shards[3].rows = 0,
                4 => invalid.weights = Some(wire::Span::default()),
                5 => invalid.shards[1].scales = None,
                _ => invalid.request_buffer.clear(),
            }
            assert!(validate_shards(&invalid, 4).is_err(), "mutation {change}");
        }
    }

    #[test]
    fn local_heads_keep_only_the_rows_a_partition_stores() {
        // Three heads per order over primes 5, 7, 11, 13, 17, 19. Rank 1 of 2 stores
        // heads 1, 3 and 5 after rank 0's 5 + 11 + 17 rows.
        let geometry = wire::EngramHash {
            method: wire::engram_hash::Method::XorModulo as i32,
            orders: vec![2, 2, 2, 3, 3, 3],
            multipliers: vec![3, 5, 7],
            primes: vec![5, 7, 11, 13, 17, 19],
            offsets: vec![0, 33, 5, 40, 16, 53],
        };
        let rows = |heads: Vec<Head>| heads.iter().map(|head| head.first_row).collect::<Vec<_>>();
        let orders = |heads: Vec<Head>| heads.iter().map(|head| head.order).collect::<Vec<_>>();
        assert_eq!(orders(local_heads(&geometry, 0, 72)), [2, 2, 2, 3, 3, 3]);
        assert_eq!(
            rows(local_heads(&geometry, 0, 33)),
            [Some(0), None, Some(5), None, Some(16), None]
        );
        assert_eq!(
            rows(local_heads(&geometry, 33, 39)),
            [None, Some(0), None, Some(7), None, Some(20)]
        );
        // An unpartitioned table stores every head at its published offset.
        assert_eq!(
            rows(local_heads(&geometry, 0, 72)),
            [Some(0), Some(33), Some(5), Some(40), Some(16), Some(53)]
        );
    }

    // The geometry of DeepSeek V4.1 Flash's layer-14 table over three ranks: primes and
    // multipliers as b12x's `build_geometry` derives them, offsets and partitions as the
    // recipe publishes them. Expected rows come from b12x's `hash_reference` equation,
    // mapped into rank 1's partition as the recipe's request graph maps them.
    #[test]
    fn published_partition_geometry_derives_the_rows_each_rank_stores() {
        let geometry = wire::EngramHash {
            method: wire::engram_hash::Method::XorModulo as i32,
            orders: (2..=4).flat_map(|order| [order; 8]).collect(),
            multipliers: vec![67716810739261, 51510806800915, 30921347202721, 82619226485591],
            primes: vec![
                16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627,
                16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769,
                16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889,
            ],
            offsets: vec![
                0, 128005432, 256010958, 16000477, 144005919, 272011457, 32000984, 160006430,
                288012030, 48001593, 176007057, 304012697, 64002262, 192007750, 320013394,
                80002973, 208008479, 336014153, 96003742, 224009260, 352014952, 112004555,
                240010079, 368015793,
            ],
        };
        let token_map: Arc<[u32]> = (0..1000u32)
            .map(|token| (token * 7919 + 13) % 99_092)
            .collect();
        // Tokens 5000 and 123456 lie outside the vocabulary.
        let rows = |first_row, stored_rows| {
            let hash = RowHash {
                token_map: token_map.clone(),
                pad: 2,
                multipliers: geometry.multipliers.clone(),
                heads: local_heads(&geometry, first_row, stored_rows),
            };
            let mut rows = Vec::new();
            hash.rows(&[998, 5000, 17], &[641, 2, 999, 123456, 77, 500], usize::MAX, &mut rows);
            rows
        };
        assert_eq!(
            rows(128005432, 128005526),
            [
                7561301, 23847752, 39203870, 63942521, 69169495, 94358013, 110226435, 116070304,
                6002299, 21986927, 37807811, 61022402, 69853420, 85989430, 98021958, 112772526,
                14573518, 26056798, 35843821, 57648189, 64403705, 82565835, 104218725, 125871246,
                11832809, 29526986, 37071280, 52257893, 64759723, 94747465, 104766103, 114710282,
                10679634, 25464059, 42911010, 63329876, 71672690, 87781799, 100152779, 125865211,
                11009230, 22395136, 44629937, 50686001, 64489818, 86911828, 106511435, 113832137,
            ]
        );
        // Every rank stores eight heads, and its rows fall inside its own partition.
        for (first_row, stored_rows) in
            [(0, 128005432), (128005432, 128005526), (256010958, 128005724)]
        {
            let rows = rows(first_row, stored_rows);
            assert_eq!(rows.len(), 6 * 8);
            assert!(rows.iter().all(|row| (0..stored_rows as i64).contains(row)));
        }
    }

    #[test]
    fn rank_selected_partitions_omit_sibling_planes_but_keep_their_rows() {
        let mut table = partitioned();
        for rank in [0, 2, 3] {
            table.shards[rank].weights = None;
            table.shards[rank].scales = None;
        }
        assert!(validate_shards(&table, 4).is_ok());
        let mut one_plane = table.clone();
        one_plane.shards[0].weights = Some(wire::Span::default());
        assert!(validate_shards(&one_plane, 4).is_err());
        let mut gap = table;
        gap.shards[2].first_row += 1;
        assert!(validate_shards(&gap, 4).is_err());
    }

    #[test]
    fn metadata_without_partitions_retains_shared_planes() {
        // EngramTable encoded before field 15 existed: rows=40, weight/scales spans.
        let bytes = [
            0x10, 0x28, 0x2a, 0x04, 0x08, 0x00, 0x10, 0x20, 0x32, 0x04, 0x08, 0x20, 0x10, 0x08,
        ];
        let table = wire::EngramTable::decode(bytes.as_slice()).unwrap();
        assert!(table.shards.is_empty());
        assert!(validate_shards(&table, 4).is_ok());
        let index = wire::ModelIndex {
            ranks: vec![wire::BundleIndex::default(); 4],
            engram: Some(wire::Engram {
                tables: vec![table],
                ..Default::default()
            }),
            ..Default::default()
        };
        for rank in 0..4 {
            let planes = spans(&index, Some(rank)).unwrap();
            assert_eq!(planes.len(), 2);
            assert!(planes.iter().all(|(owner, _)| owner.is_none()));
            assert_eq!((planes[0].1.offset, planes[0].1.nbytes), (0, 32));
            assert_eq!((planes[1].1.offset, planes[1].1.nbytes), (32, 8));
        }
    }
}
