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

## Setup

Before first use, import a Yomitan frequency dictionary to enable frequency-based filtering:

```bash
runekana --freq-dict <path-to-yomitan-dict.zip>
```

This extracts the dataset into a local SQLite cache (`XDG_STATE_HOME`). This operation only needs to be performed once.

## Usage

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