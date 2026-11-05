# Decisions

## Data source

UCI Online Retail II (dataset 502), <https://archive.ics.uci.edu/dataset/502/online+retail+ii>.
Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, confirmed on the UCI page
on 2026-11-01. Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
`make data` downloads the zip, reads both Excel sheets ("Year 2009-2010", "Year 2010-2011"),
checks columns / row count / date range, and stores one raw Parquet file in `data/raw/` (gitignored).

## Data cleaning rules (applied in this order)

Counts are produced by `python -m shopsage.data.clean` on the full download and pasted here
automatically. Each rule's count is measured *after* the rules above it.

<!-- CLEANING-RULES:START -->
| Rule | What it does | Rows affected |
|---|---|---:|
| `raw` | Rows in the downloaded file (both Excel sheets). | 1,067,371 |
| `exact_duplicates` | Identical rows on all eight columns; one copy kept. Mostly the 2010-12-01..09 overlap between the two Excel sheets. | 34,335 |
| `cancellations_to_returns` | Invoice starting with 'C' is a cancellation. Moved to the `returns` table (quantity made positive) and removed from purchases. | 19,104 |
| `bad_debt_adjustments` | Invoice starting with 'A' (accounting 'adjust bad debt' entries). Removed. | 6 |
| `non_product_codes` | Stock code does not start with five digits: POST, DOT, M, C2, D, S, BANK CHARGES, AMAZONFEE, ADJUST, gift vouchers, TEST, DCGS web-shop codes. Removed. | 4,784 |
| `non_positive_quantity` | Quantity <= 0 on a normal invoice: stock adjustments such as 'damaged', 'lost', 'found'. Removed. | 3,362 |
| `non_positive_price` | Price <= 0 (free samples, internal moves, negative adjustments). Removed. | 2,566 |
| `fully_cancelled_purchases` | Purchase line matched one-to-one to a later cancellation by the same customer with the same stock code and quantity (removes e.g. the 80,995-unit order that was cancelled the same day). Removed. | 6,239 |
| `description_conflicts` | Stock codes that appear with more than one description (count is stock codes, not rows). Each gets its most frequent description; ties broken alphabetically. | 592 |
| `missing_customer_id` | Valid product line without a Customer ID (guest checkout). Kept in `lines_all` for sales analysis; excluded from `transactions` (recommendations, RFM, CLV). | 226,637 |
| `final_transactions` | Rows in the clean `transactions` table. | 770,338 |
<!-- CLEANING-RULES:END -->

Notes:
- Descriptions are upper-cased and whitespace-normalised before choosing the one per stock code.
- Returns are kept in their own table and not subtracted from purchases (except fully
  cancelled lines, rule `fully_cancelled_purchases`). A returned item still shows interest,
  which is what the recommender learns from; RFM monetary value uses gross purchases.
- `lines_all` keeps guest-checkout lines so sales-over-time charts reflect the whole business.

## Test fixture

`tests/fixtures/raw_sample.parquet` (~30k raw rows) is built by `python -m shopsage.data.sample`:
all rows of 100 seeded-random customers active both before and after 2011-09-01, 30 other
customers, 500 guest rows, 50 non-product rows and the 'A' invoices. Redistributed under CC BY 4.0.

## Evaluation protocol

- **Split by time, never at random.** Train = everything before 2011-09-01; test = 2011-09-01 to
  2011-11-30 (three months). Validation for tuning = the last three months of train
  (2011-06-01 to 2011-08-31), with models fit on data before 2011-06-01. The final models are refit
  on the full train period before testing. 2011-12-01..09 (a partial month) is not used.
- **Warm users only.** We evaluate customers who bought something in the training period and
  something in the evaluation window; cold-start customers get popularity in the product.
- **Item catalogue = items sold in train.** Items first sold during the test window cannot be
  recommended but still count as relevant, so recall is slightly pessimistic for every model.
- **Two protocols.** `all`: every item bought in the window is relevant (repeat purchases count,
  which matches how this wholesaler's customers behave). `new`: only items the customer had never
  bought are relevant, and already-bought items are removed from the lists. `new` measures
  discovery, where "buy again" has nothing to offer (it degrades to its popularity fallback).
- **k = 10.** Relevance is binary. Model selection metric: NDCG@10.
- **Beyond accuracy.** Coverage = share of the catalogue shown to at least one user. Diversity =
  1 − mean pairwise cosine similarity (of buyer vectors) inside each list. Popularity bias = mean
  popularity percentile of recommended items (1.0 = only best-sellers).

## Models and tuning

- **Grid search on validation, every config logged to MLflow** (`make train`, experiment
  `shopsage-recsys-tuning`, local SQLite store `mlflow.db`; view with
  `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). The first grid had winners on its
  edges, so it was widened once (kNN down to 5 neighbours, ALS regularisation up to 20 and alpha
  down to 0.25). A manual probe past the new edges (512 factors, alpha 0.1, kNN 2–3 neighbours)
  gave no meaningful gain, so the grid was frozen there.
- **Item-kNN is written by hand** (cosine with shrinkage, top-n neighbours) rather than taken from
  `implicit`, because the API needs its similarity and co-purchase matrices as *evidence* for
  each recommendation.
- **ALS uses `implicit` 0.7.3** (`implicit.cpu.als.AlternatingLeastSquares`, API checked against
  the installed source: `fit(user_items)` takes a users × items CSR matrix). `num_threads=1` and
  `random_state=42` make it deterministic, so RESULTS.md is reproducible.
- **Serving models are fitted on all data** (through 2011-12-09) with the `new`-protocol
  parameters for ALS/kNN and the `all`-protocol half-life for buy-again.

## Customer analytics

- **RFM monetary = gross purchases** (returns not subtracted), recency in whole days from the
  snapshot 2011-12-10. Features are log1p-transformed (heavy right tails) and standardised before
  K-means. k is chosen by silhouette in 4–6; segment names come from centroid z-scores by a
  fixed rule (`name_segment`), so they stay meaningful if the data changes.
- **CLV library: pymc-marketing**, not `lifetimes`. `lifetimes` has had no release since 2020
  and its repository is archived; pymc-marketing is actively maintained and implements the same
  BG/NBD and Gamma-Gamma models (its utilities say they are adapted from `lifetimes`).
- **MAP fitting** instead of MCMC: deterministic, about 10 seconds, enough for point predictions.
  Full posteriors would give per-customer uncertainty and are an easy upgrade (`method="mcmc"`).
- **Gamma-Gamma fallback:** if a fit gives q ≤ 1 (seen on the small test fixture), the population
  mean spend is undefined, so expected order value falls back to the empirical mean.
- **Holdout:** calibrate up to 2011-06-09, predict the remaining 183 days. Compared with a naive
  "same rate as before" baseline.

## API and explanations

- **One service layer** (`shopsage/service.py`) loads the serving artifacts once. FastAPI, the
  assistant's tools and the MCP server all call it, so they always return the same numbers.
- **Two lists per customer.** `discover` = ALS with already-bought items removed (best model for
  new items); `buy_again` = recency-weighted re-order list (best model when repeats count).
  Unknown customers get popularity (`cold_start: true`).
- **Evidence** for every recommended item: popularity (buyers, rank), up to three of the
  customer's past purchases most similar to it (item-kNN similarity) with the true number of
  customers who bought both, and the customer's own order history with the item.
- **The LLM sees only plain-English facts built from that evidence** (not raw JSON: Mistral
  misread a popularity rank as a "similarity score" when given JSON), plus one worked example.
- **Rule-based guard** before anything is served: reject if the text names a catalogue product
  that isn't in the evidence (exact name or a distinctive word 3-gram), uses a number not in the
  evidence, or makes claims about the customer's tastes ("you enjoy", "your style", "perfect").
  Rejected or failed generations fall back to a template that restates the evidence. Results
  are cached in SQLite keyed by model + item + evidence hash.
- **Judges are configurable** (`JUDGE_MODELS`, default the `.env` model). A same-model judge is
  known to be lenient with its own output, so a second local model (gemma3) was added. The two
  disagree strongly, so human labels are the reference (`make faithfulness`).
- **Ports:** `API_PORT`/`UI_PORT` are Make variables (8000/8501 by default) because port 8000 was
  already used by another container on the development machine.

## Assistant (LangGraph) and MCP

- **Graph:** `agent` (LLM with tools bound) → `tools` (LangGraph `ToolNode`) → back to `agent`,
  plus a one-time `nudge` node when the model answers a data question without calling a tool.
- **Small-model adapters.** Mistral 7B on Ollama often writes tool calls as JSON text, which
  Ollama's parser misses. `parse_text_tool_calls` recovers them. Without this the model made no
  tool calls at all and invented answers with fake citations.
- **Deterministic guards around the LLM.** (1) A scope rule refuses write requests and
  personal-data requests before the LLM is called. (2) The SQL tool is read-only by design: sqlglot
  allows a single SELECT over whitelisted tables, with no file-reading or table functions, a forced
  LIMIT ≤ 100, a read-only DuckDB connection and a 5 s timeout. (3) A grounding check on the final
  answer: if it contains a number that isn't in any tool result (or the question), the answer is
  replaced by the raw tool output.
- **Same tools over MCP** (`mcp` 2.3: `mcp.server.mcpserver.MCPServer`, the 2.x rename of
  `FastMCP`), so any MCP client gets the identical functions.
- **Evaluation** compares with the same LLM without tools and reports both the model's draft
  and the served (guarded) answer, so the guard's effect is visible rather than hidden.
