# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 Small NVFP4 preparation on four GPUs with bundled DSpark and vision."""

from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, Speculation

from .frontend import SAMPLING_DEFAULTS


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("Inkling 2 preparation requires four GPUs on one host")
    if plan.prompt_logprobs:
        raise ValueError("Inkling 2 preparation does not capture prompt scoring graphs")
    speculation = plan.speculation
    if speculation is not None and (speculation.draft_head_quantization is not None
                                   or speculation.markov_head_quantization is not None):
        raise ValueError("Inkling 2 retains the bundled draft checkpoint precision")
    drafting = {} if speculation is None else dict(speculative_widths=speculation.widths,
        speculative_fixed_width=speculation.fixed_width, speculative_token_capacity=plan.speculative_token_capacity,
        speculative_graph_budget=speculation.graph_budget or 64)
    config = PreparationConfig(collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        compile_headroom_bytes=plan.compile_headroom_bytes,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)), context=plan.context, prefill_chunk_size=plan.prefill_chunk,
        max_active=plan.concurrency, image_concurrency=plan.image_concurrency,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, kernel_cache=kernel_cache,
        drafting=speculation is not None, vision=not plan.language_model_only,
        mtp_only=speculation is not None and speculation.mtp_only,
        benchmark_selection=plan.benchmark_selection, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(name="inkling2", served_model_name="Inkling-2-Small-NVFP4", prepare=prepare, usage=usage,
    sampling_defaults=SAMPLING_DEFAULTS,
    checkpoints=(Checkpoint(alias="inkling-2-small-nvfp4", repository="/data/models/Inkling-2-Small-NVFP4",
        revision="local", architecture="Inkling2ForConditionalGeneration", stored_bytes=344_366_116_864,
        device_bytes=344_366_116_864, expansion_bytes=5_673_717_760, draft_bytes=2_684_203_392),),
    context=1048576, page_tokens=128, concurrency=8, max_concurrency=8, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=7, token_capacity=64, packed_verification=True, graph_budget=64),
    max_draft_tokens=7, max_images=4, image_bytes=32 << 20, verification_rows=0, language_model_only=True,
    ranks=MappingProxyType({4: RankCosts(31_293_440, 655_360, 5 << 30)}),
    plain_ranks=MappingProxyType({4: RankCosts(26_935_296, 524_288, 5 << 30)}))
