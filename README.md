# runekana

A furigana (ruby) annotation tool for Japanese EPUB files.

`runekana` injects furigana into Japanese text while preserving XHTML document structure. It combines morphological analysis (Sudachi) with optional LLM-based verification to resolve ambiguous readings and complex orthography.

## Features

- **Structural Integrity**: Preserves existing ruby, HTML tags, and metadata during XHTML parsing.
- **Precise Tokenization**: Uses `sudachipy` for morphological analysis.
- **Frequency Filtering**: Supports Yomitan dictionaries to skip common vocabulary.
- **LLM Verification**: Context-aware correction of ambiguous readings via Gemini or OpenAI.
- **Persistent Caching**: SQLite-backed frequency index and TSV caching for API efficiency.

## Installation

Requires Python 3.13+.

```bash
uv tool install .
```

Or via Nix:
```bash
nix profile install .
```

Or run via container:
```bash
podman run -it --rm \
  -v $(pwd):/data:Z \
  --env GEMINI_API_KEY \
  ghcr.io/attrr/runekana:latest \
  /data/input.epub /data/output.epub --verify
```

## Setup

Before first use, import a Yomitan frequency dictionary to enable frequency-based filtering:

```bash
runekana --freq-dict <path-to-yomitan-dict.zip>
```

This extracts the dataset into a local SQLite cache (`XDG_STATE_HOME`). This operation only needs to be performed once.

## Usage
<details>
<summary>runekana --help</summary>

```text
usage: runekana [-h] [--skip-top SKIP_TOP] [--freq-dict FREQ_DICT] [--dict DICT] [--verify] [--contextual]
                [--provider {gemini,openai}] [--model MODEL] [--base-url BASE_URL] [--canary-url CANARY_URL]
                [--concurrency CONCURRENCY] [--batch-size BATCH_SIZE] [--generated-dir GENERATED_DIR]
                [--price-input PRICE_INPUT] [--price-output PRICE_OUTPUT] [--verbose]
                input output

Add furigana annotations to Japanese EPUB files.

positional arguments:
  input                 Input EPUB file
  output                Output EPUB file

options:
  -h, --help            show this help message and exit
  --skip-top SKIP_TOP   Skip the top N most frequent Japanese words (default: 1500).
  --freq-dict FREQ_DICT
                        Path to Yomitan frequency dictionary (ZIP or folder) to import into local cache.
  --dict DICT           Local dictionary file (TSV: word<TAB>reading).
  --verify              Verify readings with LLM (needs GCP_PROJECT for gemini, or OPENAI_API_KEY for openai).
  --contextual          Deduplicate candidates by context (sends identical words in different contexts to LLM).
  --provider {gemini,openai}
                        API provider for verification (default: gemini).
  --model MODEL         LLM model name to use for verification.
  --base-url BASE_URL   Custom base URL for OpenAI/Gemini-compatible providers (e.g. DeepSeek, vLLM).
  --canary-url CANARY_URL
                        URL to check for internet connectivity (default: Google 204). Try Cloudflare or GrapheneOS one if
                        blocked.
  --concurrency CONCURRENCY
                        Max parallel LLM requests during verification (default: 5). Lower for rate-limited APIs.
  --batch-size BATCH_SIZE
                        Number of words to send in a single LLM request (default: 100). Lower for faster feedback.
  --generated-dir GENERATED_DIR
                        Directory to save LLM outputs as JSON.
  --price-input PRICE_INPUT
                        Price per 1M input tokens (USD). Used for cost estimation.
  --price-output PRICE_OUTPUT
                        Price per 1M output tokens (USD). Used for cost estimation.
  --verbose, -v         Increase verbosity (-v, -vvv)
```
</details>

Basic annotation:

```bash
runekana input.epub output.epub
```

### Advanced Usage

Verify ambiguous readings using an LLM (requires authentication):

```bash
runekana input.epub output.epub \
  --verify \
  --contextual \
  --provider gemini \
  --model gemini-3.1-flash-lite-preview
```

*Note: The `--contextual` flag passes surrounding text to the LLM for higher accuracy on context-dependent vocabulary.*

Adjust annotation density by changing the frequency threshold (default is 1500):

```bash
runekana input.epub output.epub --skip-top 2000
```

## Configuration

For LLM verification, expose the relevant environment variables:

**Google Gemini**
```bash
export GEMINI_API_KEY="your-api-key"
```
*Alternatively, for GCP Vertex AI:*
```bash
export GCP_PROJECT="your-project-id"
export GCP_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

**OpenAI**
```bash
export OPENAI_API_KEY="your-api-key"
```