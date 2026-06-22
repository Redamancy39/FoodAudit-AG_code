# Category anchoring specification

This note documents the category-anchoring implementation used by the FoodAudit-AG production-backend evaluation.

## Candidate generation

1. Load every `FoodCategory` pair (`code`, `name`) from Neo4j.
2. Encode category names and the extracted product name with `BAAI/bge-small-zh-v1.5`.
3. Compute cosine similarity between the product vector and all category-name vectors in the application layer.
4. Retain the 10 highest-scoring categories in retrieval order.

The system does not use a Neo4j vector index. It does not apply a minimum similarity threshold. Cosine similarity is the only numeric retrieval feature; rule availability and downstream verdicts do not affect anchor ranking.

## Semantic selection

The ordered candidates are formatted as `<name> (Code: <code>)` and supplied to Qwen3-Max with temperature 0.1. The prompt asks the model to select one candidate, normalize commercial wording to a product type, and use the nearest reasonable parent when no exact category exists.

The returned `target_code` is resolved first against the standardized category table. Exact `target_name` matching is used only if the code does not resolve, because category names are not guaranteed to be unique. If neither field resolves, the production backend reports a mapping failure.

No explicit secondary rule is applied to exact cosine-score ties: tied entries retain the retrieval library's order, and the model makes the final semantic selection. There are no learned weights, hand-tuned composite scores, or rule-support tie breakers.

## Evaluation-harness distinction

The ablation evaluator includes a vector-Top-1 fallback when the reranker output is invalid or the call fails. That fallback is specific to the ablation harness. The production backend used by `evaluate_foodaudit.py` returns a mapping failure instead and generated the reported FoodAudit-AG backend results.
