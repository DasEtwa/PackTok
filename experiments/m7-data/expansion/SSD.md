# Historical-corpus HDD and SSD comparison

The follow-up authorizes a 10,000,000,000-byte floor for free SSD space. Windows identifies D: as the WDC WD20EZAZ HDD and C: as the SanDisk SSD PLUS 120 GB. Ubuntu-24.04's native VHDX, source corpus, benchmark input cache and final JSONL are all on D:. Only the temporary deduplication files move between physical drives.

The selected input contains 1,030 admitted whole documents and 296,724,588 UTF-8 text bytes from `dataset-eval-v1`. The sampler uses source ID order and independent per-language and per-split caps. It checks the size and SHA-256 of every cached document against the verified historical catalog. The comparison reruns the Rust M7 build with identical input, output placement, grouping, language filters and dedup logic. [Benchmark runner](ssd-pilot.py).

The initial direct SSD index on WSL's NTFS interface had written 115,422,726 bytes to staged shards after 649 seconds. The interrupted run yielded no completed build or SSD/HDD wall-time ratio. The follow-up uses two temporary 1.5 GiB loop images. They use the same ext4 formatting and mount options, with one image on each Windows volume. Both index paths then have equivalent Linux filesystem semantics. The image setup and source-cache warm-up are outside the per-build elapsed time. [Direct SSD observation](ssd-direct-intermediate.json), [loop mount wrapper](ssd-loop-pilot.sh).

The 10 GB floor is enforced against the physical SSD backing mount throughout each run. Image preparation also checks that the full image size fits above that floor. The wrapper mounts scratch volumes as root in the authorized Ubuntu-24.04 distribution and executes PackTok as the supplied unprivileged user. Its cleanup uses the two private mount directories and image files created for the run.

## Result

| Index backing | Build and verify | User + system CPU | Peak RSS | Sampled peak index | Lowest sampled SSD free space |
|---|---:|---:|---:|---:|---:|
| D: HDD | 166.06 s | 160.42 s | 70,840 KiB | 269,488,128 B | 18,140,651,520 B |
| C: SSD | 173.55 s | 163.53 s | 70,648 KiB | 269,488,128 B | 18,139,807,744 B |

The SSD run took 7.49 s longer, a difference of 4.51% in this single pair. CPU time accounted for about 94–97% of single-core wall time. Both runs verified 6 shards, 1,030 documents and 296,724,588 text bytes. The dataset and source catalog manifests were byte-identical, and the per-shard hashes were verified in both runs. The result gives no measured SSD speedup for this workload. The variance across repetitions and the effect under the sustained I/O load of the full 1.45 GB build remain unmeasured. [Measured receipt](ssd-loop-results.json), [verification and cleanup](ssd-loop-verification.txt).

The independent Gemini 4 Hydrogen Medium review highlighted the warm-cache scope, the single-HDD timing and the difference between a storage-medium benchmark and a filesystem-path benchmark. The complete comparison holds ext4 formatting and mount options constant. The earlier direct NTFS attempt remains a separate intermediate observation. An initial wrapper launch was rejected by the argument parser before the benchmark, and its images were unmounted and removed. The argument array was corrected before the complete pair.

## Reproduction

Use a native Ubuntu-24.04 copy of these scripts. Put the benchmark work and output directories on native WSL storage backed by D:. The paths `$sample` and `$work` must be new. The mount wrapper requires root inside that distribution.

```sh
python3 ssd-pilot.py "$bin" "$dataset" "$sample" /mnt/d /mnt/c --prepare-only
sudo bash ssd-loop-pilot.sh "$USER" "$bin" "$dataset" "$sample" "$work" /mnt/d "$ssd_temp_parent"
```

The experiment uses warm source data, fresh indexes and no OS cache drops. The storage comparison includes each normal M7 build and full shard verification. Because the pipeline's redb index is disposable, filesystem caching and the kernel's writeback policy can absorb a portion of its writes. The result characterizes this historical subset and software path. It is separate from a sustained physical-device throughput benchmark.
