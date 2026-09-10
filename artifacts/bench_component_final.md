# FastWAM Components

| Component | 10 steps | 1 step | 1 step compile | 1 step compile async | 1 step compile async kernel |
| --- | --- | --- | --- | --- | --- |
| vae_encode | 4.974 ms | 5.331 ms | 3.981 ms | 3.991 ms | 3.997 ms |
| text_encode | 17.579 ms | 16.929 ms | 16.987 ms | 16.993 ms | 16.918 ms |
| video_prepare | 0.511 ms | 0.488 ms | 0.166 ms | 0.167 ms | 0.167 ms |
| video_dit_prefill | 26.270 ms | 25.329 ms | 11.652 ms | 13.835 ms | 7.959 ms |
| action_dit_denoise | 274.860 ms | 26.841 ms | 8.534 ms | 12.781 ms | 6.814 ms |
| action_scheduler | 0.246 ms | 0.024 ms | 0.008 ms | 0.008 ms | 0.009 ms |
| overlap_correction | 0.000 ms | 0.000 ms | 0.000 ms | -12.600 ms | -6.593 ms |
| other | 8.059 ms | 7.258 ms | 0.390 ms | 0.466 ms | 0.472 ms |
| total | 332.499 ms | 82.201 ms | 41.717 ms | 35.641 ms | 29.744 ms |
| total_uninstrumented | 328.710 ms | 81.635 ms | 41.638 ms | 35.307 ms | 29.381 ms |

Component CUDA intervals are measured in a separate instrumented pass. overlap_correction removes double-counted concurrent time; other is profiled wall time minus interval union. Components + overlap_correction + other = total. Use total_uninstrumented for latency comparisons. All totals include VAE. Text encoding per request: True. Text is not compiled. Setup and validation are excluded.
