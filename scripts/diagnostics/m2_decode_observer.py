"""Read-only sampler instrumentation, installed via vLLM's collective_rpc API."""

from __future__ import annotations


def install_observer(worker, path):
    import json

    runner = worker.model_runner
    if hasattr(worker, "_fca_observer"):
        raise RuntimeError("Observer already installed")
    handle = open(path, "x")  # noqa: SIM115 - closed by remove_observer across RPCs
    pending = {}
    count = [0]

    def before(module, args, kwargs):
        logits = kwargs["logits"]
        batch = runner.input_batch
        ids = list(batch.req_ids)
        assert len(ids) == logits.shape[0]
        # Allocate separate storage. No write to input logits or engine state.
        pending["logits"] = logits.detach().float().clone()
        pending["dtype"] = str(logits.dtype)
        pending["requests"] = []
        for i, request_id in enumerate(ids):
            start, end = int(runner.query_start_loc_np[i]), int(runner.query_start_loc_np[i + 1])
            assert end > start
            pending["requests"].append(
                {
                    "request_id": request_id,
                    "prompt_length": int(batch.num_prompt_tokens[i]),
                    "output_prefix": list(batch.req_output_token_ids[i]),
                    "computed_before": int(batch.num_computed_tokens_cpu[i]),
                    "scheduled_tokens": end - start,
                    "last_input_position": int(runner.positions_np[end - 1]),
                }
            )
        return None

    def after(module, args, kwargs, output):
        logits = pending.pop("logits")
        lp = logits.log_softmax(dim=-1)
        tokens = output.sampled_token_ids.flatten().tolist()
        rows = pending.pop("requests")
        assert len(tokens) == len(rows)
        for i, (row, token) in enumerate(zip(rows, tokens, strict=True)):
            row.update(
                token_id=token,
                logprob=float(lp[i, token]),
                target_logit=float(logits[i, token]),
                logsumexp=float(logits[i].logsumexp(dim=-1)),
                rank=int((lp[i] >= lp[i, token]).sum()),
            )
        handle.write(
            json.dumps(
                {"call_index": count[0], "dtype": pending["dtype"], "rows": rows}, allow_nan=False
            )
            + "\n"
        )
        handle.flush()
        count[0] += 1
        return None

    pre = runner.sampler.register_forward_pre_hook(before, with_kwargs=True)
    post = runner.sampler.register_forward_hook(after, with_kwargs=True)
    worker._fca_observer = (pre, post, handle, count)
    return {"status": "INSTALLED", "mutation": False}


def remove_observer(worker):
    pre, post, handle, count = worker._fca_observer
    pre.remove()
    post.remove()
    handle.close()
    del worker._fca_observer
    return {"status": "REMOVED", "sampler_calls": count[0]}
