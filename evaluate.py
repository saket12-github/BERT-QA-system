import argparse
import json
from statistics import mean
from time import perf_counter

from datasets import load_dataset

from metrics import best_ground_truth_metric, exact_match_score, f1_score
from qa_engine import MODEL_NAME, QAEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA model on SQuAD 2.0.")
    parser.add_argument("--split", default="validation", help="Dataset split to use.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of samples to evaluate for quick benchmarking.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset("squad_v2", split=args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    engine = QAEngine(model_name=MODEL_NAME)

    em_scores = []
    f1_scores = []
    latencies = []

    run_start = perf_counter()
    for row in dataset:
        pred = engine.predict(context=row["context"], question=row["question"])
        prediction = pred["answer"] if not pred["no_answer"] else ""
        references = row["answers"]["text"]

        em_scores.append(
            best_ground_truth_metric(prediction, references, exact_match_score)
        )
        f1_scores.append(best_ground_truth_metric(prediction, references, f1_score))
        latencies.append(pred["latency_ms"])

    total_time = perf_counter() - run_start
    results = {
        "model_name": MODEL_NAME,
        "split": args.split,
        "samples_evaluated": len(em_scores),
        "exact_match": round(mean(em_scores), 4) if em_scores else 0.0,
        "f1": round(mean(f1_scores), 4) if f1_scores else 0.0,
        "avg_latency_ms": round(mean(latencies), 2) if latencies else 0.0,
        "throughput_samples_per_sec": round(len(em_scores) / total_time, 2)
        if total_time > 0
        else 0.0,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved evaluation report to: {args.output}")


if __name__ == "__main__":
    main()
