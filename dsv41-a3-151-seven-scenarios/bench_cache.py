"""Use SGLang's serving measurement with page-aligned warmed prefixes."""

import os
from concurrent.futures import ThreadPoolExecutor

import requests
from sglang.benchmark import serving


native_dataset = serving.get_dataset


def get_dataset(args, tokenizer, model_id):
    prefix_len = int(os.environ["BENCH_PREFIX_TOKENS"])
    dp_size = int(os.environ["BENCH_ATTN_DP"])
    rows = native_dataset(args, tokenizer, model_id)
    assert len(rows) == args.num_prompts
    assert all(len(row.prompt) == args.random_input_len for row in rows)
    base_url = f"http://{args.host}:{args.port}"
    flush = requests.post(f"{base_url}/flush_cache", params={"timeout": 30}, timeout=60)
    flush.raise_for_status()
    prefix = rows[0].prompt[:prefix_len]

    def warm(rank):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_ids": prefix + [prefix[0]],
                "cache_salt": args.tag,
                "routed_dp_rank": rank,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            },
            timeout=900,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        assert meta["prompt_tokens"] == prefix_len + 1 and meta["completion_tokens"] == 1, meta
        print(f"Warmed {prefix_len} tokens on Attention DP rank {rank}", flush=True)

    if prefix:
        with ThreadPoolExecutor(max_workers=min(dp_size, len(rows))) as pool:
            list(pool.map(warm, range(min(dp_size, len(rows)))))

    for index, row in enumerate(rows):
        row.prompt = prefix + row.prompt[prefix_len:]
        row.extra_request_body["cache_salt"] = args.tag if prefix else f"{args.tag}-{index}"
        row.extra_request_body["routed_dp_rank"] = index % dp_size
    return rows


if __name__ == "__main__":
    serving.get_dataset = get_dataset
    serving.cli_main()
