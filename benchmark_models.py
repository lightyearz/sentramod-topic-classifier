"""
Lightweight benchmarking script to compare ONNX inference latency vs an LLM/BART (if available).

Usage:
  python services/topic-classifier-service/benchmark_models.py --runs 20

The script warms up each model then measures N runs and prints mean, median, stddev, and percentiles.
"""

import os
import sys
import time
import statistics
import argparse
import json

from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    # Use the application classifier init (handles tokenizer fallbacks & ONNX/BART selection)
    from app.application.classifier import get_classifier
except Exception:
    get_classifier = None


def load_settings():
    script_dir = os.path.dirname(__file__)
    service_app_dir = os.path.abspath(os.path.join(script_dir, "app"))
    if service_app_dir not in sys.path:
        sys.path.insert(0, service_app_dir)
    try:
        from config import settings

        return settings
    except Exception:

        class Dummy:
            ONNX_MODEL = "./services/topic-classifier-service/models/deberta-onnx"
            BART_MODEL = "valhalla/distilbart-mnli-12-1"

        return Dummy()


def _invoke_pipeline_with_timeout(pipeline_obj, sentence, candidate_labels, timeout=15):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(pipeline_obj, sentence, candidate_labels, multi_label=True)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeoutError:
            fut.cancel()
            raise TimeoutError("Pipeline call exceeded timeout")


def benchmark_pipeline(
    pipeline_obj, sentence, candidate_labels, runs=20, warmup=2, timeout=15
):
    # warmup
    for _ in range(warmup):
        pipeline_obj(sentence, candidate_labels, multi_label=True)

    times = []
    for i in range(1, runs + 1):
        t0 = time.perf_counter()
        try:
            _ = _invoke_pipeline_with_timeout(
                pipeline_obj, sentence, candidate_labels, timeout=timeout
            )
        except Exception as e:
            print(f"⚠️ Run {i}/{runs} failed: {e}")
            continue
        times.append((time.perf_counter() - t0) * 1000.0)
        if i % max(1, runs // 5) == 0:
            print(f"   ✅ Completed {i}/{runs} runs")

    return times


def fmt_stats(times):
    return {
        "count": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "p90_ms": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else None,
    }


def run_bench(
    onx_model_path, bart_model, runs=20, skip_llm=False, allow_download=False
):
    sentence = "I am worried about my friend. They have been sad and it's affecting their school."
    candidate_labels = [
        "suicide",
        "self-harm",
        "mental health",
        "abuse",
        "dating & relationships",
        "academic performance",
    ]

    results = {}

    # ONNX - prefer to use the application's classifier instance which implements tokenizer fallbacks
    if get_classifier is None:
        results["onnx"] = {
            "error": "get_classifier not importable; run this from the service project root"
        }
    else:
        try:
            classifier_instance = get_classifier()
            if getattr(classifier_instance, "method", None) != "onnx":
                results["onnx"] = {
                    "error": f"Classifier method is {classifier_instance.method}; expected 'onnx'"
                }
            else:
                pipe = classifier_instance.classifier
                times = benchmark_pipeline(pipe, sentence, candidate_labels, runs=runs)
                results["onnx"] = fmt_stats(times)
        except Exception as e:
            results["onnx"] = {"error": str(e)}

    # BART/LLM
    if skip_llm:
        results["llm"] = {"skipped": True}
    else:
        try:
            if not allow_download and not os.path.exists(bart_model):
                # bart_model could be a remote HF repo name; skip if local path is not found
                results["llm"] = {
                    "skipped": "local path not found, set --allow-download to fetch"
                }
            else:
                bart_pipe = pipeline(
                    "zero-shot-classification",
                    model=bart_model,
                    device=-1,
                    local_files_only=not allow_download,
                )
                times = benchmark_pipeline(
                    bart_pipe, sentence, candidate_labels, runs=runs
                )
                results["llm"] = fmt_stats(times)
        except Exception as e:
            results["llm"] = {"error": str(e)}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of timed runs to average"
    )
    parser.add_argument(
        "--allow-llm",
        action="store_true",
        default=False,
        help="Allow LLM/BART benchmarking (will attempt to load or download)",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        default=False,
        help="Allow downloading missing models from the hub for LLM benchmarking",
    )
    args = parser.parse_args()

    settings = load_settings()
    onnx_path = getattr(
        settings,
        "ONNX_MODEL",
        "./services/topic-classifier-service/models/deberta-onnx",
    )
    bart_model = getattr(settings, "BART_MODEL", "valhalla/distilbart-mnli-12-1")

    print("Benchmarking: ONNX path:", onnx_path)
    print("Benchmarking: LLM/BART model:", bart_model)
    run_bench(
        onnx_path,
        bart_model,
        runs=args.runs,
        skip_llm=not args.allow_llm,
        allow_download=args.allow_download,
    )
