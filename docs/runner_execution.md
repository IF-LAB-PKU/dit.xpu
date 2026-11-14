# Runner profiling and shared CLIP execution

## Profiling

Set `SD_RUNNER_PROFILE=1` to collect host spans for a complete runner call,
including its graph-cut segments. One `runner_profile` line is emitted on
return, including failures. Profiling is off by default; disabled span timers
do not read the clock. Durations use microseconds.

The exclusive spans are graph/segment construction (`build`), graph-cut
planning (`plan`), packed-weight registration/validation (`packed_prepare`),
packed-weight staging (`staging`), parameter offload, allocator reserve,
graph allocation/binding (`allocate`), input copying, the backend compute
call, cache-output copying, result readback and the final release/restore.
`other` contains the remaining host work, including routing-buffer planning,
cache transactions and segment cleanup outside those spans. Backend planning
is included in `compute`. `reserves` counts actual calls, including the second
reserve in the multi-buffer allocation path; it is not a reallocation count.

HTP additionally emits one `htp_dispatch_profile` line per backend graph:

- `calls`: completed message-channel dispatches, including failed responses.
- `submit_us`: operation preparation/mapping until the DSP doorbell.
- `wait_us`: doorbell-following wait until completion is observed; this includes
  DSP execution and any overlapping CPU epilogue work.
- `dsp_reported_us` and `dsp_reported_calls`: kernel durations for supported
  response formats (native Linear epilogues, convolution, attention and aux).
  Unreported kernels are excluded. This is not a complete DSP timeline.

Dispatch spans are **inside** the runner's compute span, and kernel durations
are inside the dispatch wait. Do not add these levels together. Wait minus
reported kernel time is not necessarily idle host time: it may also include
DSP allocations, protocol work, unsupported timing formats and scheduling.
For asynchronous OpenCL paths, compute measures the backend call rather than
forcing completion; later waits are charged where they occur. Profiling does
not insert a synchronization or alter retirement rules.

## One CLIP pass for two SD3 outputs

The SD3 conditioner needs token hidden states from the selected `clip_skip`
layer and a pooled vector from the final layer. Previously, it executed the
CLIP encoder separately for each output. The first prompt chunk now captures
the selected intermediate, continues to the final layer and produces both
outputs from one traversal. Later chunks keep the existing hidden-only path.
Token weights are still applied only to hidden states, and EOS selection,
final normalization and text projection keep their existing semantics.

`SD_CLIP_SHARED_PASS=0` restores separate passes; reuse is enabled by default
in the SD3 conditioner and works with both CPU and native HTP routing. It does
not cache embeddings between requests or retain model weights. Existing
single-output CLIP callers keep their API and execution path.

The combined graph expands the final pooled branch before its final consumer
concatenates the hidden/pooled outputs for one runner readback. The captured
intermediate retains its graph-cut name and stays live through the final
consumer, including across segments. Only host tensors are returned; graph
pointers never survive the request. The bounded 77-token combined path adds
a small temporary output copy. Inputs longer than the model context or
unusual `clip_skip` values fall back to the separate-pass implementation.

## Device verification

Use the matched device runtime and the existing text test:

```sh
SD_CLIP_SHARED_PASS_AB=1 ./test-text-encoder clip_g MODEL htp 8 OUTPUT_PREFIX
SD_CLIP_SHARED_PASS_AB=1 ./test-text-encoder clip_l ORIGINAL_MODEL cpu 2 OUTPUT_PREFIX
SD_CLIP_SHARED_PASS_AB=1 SD_TEST_CLIP_SKIP_CYCLE=1 \
  ./test-text-encoder clip_g MODEL htp 6 OUTPUT_PREFIX ""
```

Normal A/B order is separate/shared/shared/separate, repeated. The cycle mode
compares separate/shared pairs at `clip_skip` 2, 1 and the layer count, including
an early intermediate that must survive subsequent graph cuts. The test
asserts byte-identical hidden and pooled outputs against the separate-pass
reference unless the independent fusion A/B test is also enabled.

Build `sd-cli`, `test-text-encoder` and `test-graph-cut-views` with the normal
Android HTP configuration. Full-generation validation must also compare the
final images under identical prompts, seeds and model files.

## T5 packed-weight prefetch

`SD_HTP_PACKED_PREFETCH=1` enables one-segment lookahead for graph-cut T5
with deferred, offline-packed F16 weights. Other runners and non-deferred
weights retain synchronous staging. The sparse/OpenCL path is excluded.
The switch is off by default because the extra bank increases memory usage.

The main thread plans each segment's tensors and session owners, initializes
loader metadata, and reserves a private next bank. One background task reads
copied names/types/sizes into raw reserved addresses while the current bank
is consumed. It never changes tensor bindings or backend mappings. Before
the next segment, the main thread joins the reader, checks the tensor set,
binds the completed bank, then reuses the released previous bank. All exits
drain the reader before freeing reservations. Reservation/launch failure
falls back to synchronous staging; an actual read failure fails the request.

Each bank independently obeys `SD_HTP_PACKED_STAGING_BUDGET_MB` and
`SD_HTP_PACKED_STAGING_SLOT_MB`. The default 512 MiB bank budget therefore
permits **up to 1024 MiB total staging allocation**, excluding model parameters,
activations and backend memory. Graph-cut sizing remains at the original
single-bank budget. Physical slot session affinity remains unchanged; a
prefetched slot is mapped only when consumed. The spare bank is freed at
request exit; the normal active bank retains its existing lifecycle.

`htp_prefetch_profile` reports started/skipped tasks, host join wait,
background load duration for taken batches and peak allocated bytes across
both banks. Cancellation drain time is outside this join counter. Load
duration overlaps runner spans and must not be added to them. `staging_us`
includes host reservation/launch and remaining join wait, rather than the
full overlapping background read time. Peak bank bytes are allocation
accounting, not process RSS or DSP mapping usage.

```sh
SD_RUNNER_PROFILE=1 SD_HTP_PREFETCH_AB=1 ./test-text-encoder \
  t5xxl PACKED_MODEL htp 8 OUTPUT_PREFIX
```

This uses synchronous/prefetch/prefetch/synchronous order twice and checks
every encoded output byte against the first run. `test-htp-packed-staging`
also covers gated concurrent loading, stale generations, owner affinity,
capacity failure, reader failure/exception, partial binding rollback and
destruction with a reader in flight.
