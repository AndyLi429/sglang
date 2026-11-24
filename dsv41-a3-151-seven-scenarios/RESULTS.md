# 151 · A3 单机混部 7 个场景

每行选该场景目前已核验、满足平均 SLA 的最高输出吞吐单次跑测。所有行均为 DeepSeek V4.1 Flash W4A8，卡数按服务可见的 16 个逻辑 NPU 计。

TTFT、TPOT 和 E2E 均为单请求平均值。E2E 从原始结果的 `mean_e2e_latency_ms` 读取，整轮压测耗时另见 [metrics.csv](metrics.csv)。

1M 输入为 1,000,000 token，实际命中 98.9952%。64K 和 128K 两种 90% 档实际均为 90.0391%。

16K/1K、20 ms 的 C13 行为 seed 12346 的达标单次结果。同配置另一 seed 的平均 TPOT 为 20.06 ms。

| 场景 | 并发 | 卡数 | 启动命令 | 测试命令 | 性能截图 | TTFT | TPOT | E2E 时间 | 输出吞吐 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 16K/1K · 0% · <30 s / <50 ms | 56 | 16 | [launch.sh](01-16k-cold-50ms/launch.sh) | [bench.sh](01-16k-cold-50ms/bench.sh) | [黑屏 PNG](01-16k-cold-50ms/bench.black.png) | 18.39 s | 37.20 ms | 56.45 s | 983.95 tok/s |
| 16K/1K · 0% · <10 s / <20 ms | 13 | 16 | [launch.sh](02-16k-cold-20ms/launch.sh) | [bench.sh](02-16k-cold-20ms/bench.sh) | [黑屏 PNG](02-16k-cold-20ms/bench.black.png) | 6.78 s | 18.58 ms | 25.79 s | 513.92 tok/s |
| 64K/1K · 0% · <30 s / <50 ms | 12 | 16 | [launch.sh](03-64k-cold-50ms/launch.sh) | [bench.sh](03-64k-cold-50ms/bench.sh) | [黑屏 PNG](03-64k-cold-50ms/bench.black.png) | 27.21 s | 25.13 ms | 52.92 s | 221.43 tok/s |
| 64K/1K · 90% · <30 s / <50 ms | 64 | 16 | [launch.sh](04-64k-cache90-50ms/launch.sh) | [bench.sh](04-64k-cache90-50ms/bench.sh) | [黑屏 PNG](04-64k-cache90-50ms/bench.black.png) | 10.33 s | 21.47 ms | 32.30 s | 1,715.31 tok/s |
| 128K/1K · 90% · <30 s / <50 ms | 40 | 16 | [launch.sh](05-128k-cache90-50ms/launch.sh) | [bench.sh](05-128k-cache90-50ms/bench.sh) | [黑屏 PNG](05-128k-cache90-50ms/bench.black.png) | 14.50 s | 34.02 ms | 49.30 s | 787.27 tok/s |
| 128K/1K · 90% · <10 s / <20 ms | 8 | 16 | [launch.sh](06-128k-cache90-20ms/launch.sh) | [bench.sh](06-128k-cache90-20ms/bench.sh) | [黑屏 PNG](06-128k-cache90-20ms/bench.black.png) | 7.69 s | 15.28 ms | 23.32 s | 344.28 tok/s |
| 1M/1K · 99% · <30 s / <50 ms | 16 | 16 | [launch.sh](07-1m-cache99-50ms/launch.sh) | [bench.sh](07-1m-cache99-50ms/bench.sh) | [黑屏 PNG](07-1m-cache99-50ms/bench.black.png) | 21.51 s | 39.92 ms | 62.35 s | 257.17 tok/s |

## 复现

进入 151 的 `lph-1118` 容器，每次只运行一个场景。在一个终端执行对应的 `bash launch.sh`，服务就绪后在另一终端执行对应的 `bash bench.sh`。
每份 Shell 都是实际跑测使用的完整文件。启动脚本使用容器里已部署的环境、代码运行树、算子库和权重路径。压测脚本保留原客户端参数和原始输出路径。包内的 [bench_cache.py](bench_cache.py) 是原压测客户端的副本。

每个场景目录附原始终端输出 `bench.log`。黑屏 PNG 从该日志生成，已经校验日志 SHA-256，并核对表中指标与对应日志相符。
