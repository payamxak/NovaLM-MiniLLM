"""Fetch short Wikipedia page extracts with revision-level attribution metadata."""
import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def safe_stem(title: str) -> str:
    stem = re.sub(r"[^\w.-]+", "_", title, flags=re.UNICODE).strip("_.")
    return stem[:100] or "article"


def fetch_page(language: str, title: str) -> tuple[dict, dict]:
    encoded_title = quote(title.replace(" ", "_"), safe="")
    headers = {"User-Agent": "NovaLM-local-research/0.1 (data smoke collector)"}

    def get_json(url):
        request = Request(url, headers=headers)
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    page_url = f"https://{language}.wikipedia.org/w/rest.php/v1/page/{encoded_title}/with_html"
    summary_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
    page, summary = get_json(page_url), get_json(summary_url)
    if not page.get("license", {}).get("url"):
        raise ValueError(f"Wikipedia page {title!r} did not provide a text license")
    if not summary.get("extract", "").strip():
        raise ValueError(f"Wikipedia summary for {title!r} contains no extract")
    return page, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True,
                        help="Wikipedia language code, for example en or fa")
    parser.add_argument("--title", action="append", required=True,
                        help="Article title; repeat to fetch multiple pages")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--license-declaration",
                        help="Optional local declaration; the API-reported page license is recorded separately")
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,11}", args.language):
        parser.error("language must be a short Wikipedia language code")
    if args.delay_seconds < 0:
        parser.error("delay-seconds cannot be negative")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    used_names = set()
    for index, requested_title in enumerate(args.title):
        try:
            page, summary = fetch_page(args.language, requested_title)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            parser.error(f"failed to fetch {requested_title!r}: {exc}")
        title = page.get("title", requested_title)
        stem = safe_stem(title)
        if stem.casefold() in used_names:
            stem = f"{stem}_{page.get('pageid', index)}"
        used_names.add(stem.casefold())
        local_path = output / f"{stem}.txt"
        local_path.write_text(summary["extract"].strip() + "\n", encoding="utf-8")
        source_url = summary.get("content_urls", {}).get("desktop", {}).get("page")
        if not source_url:
            source_url = f"https://{args.language}.wikipedia.org/wiki/{quote(page.get('key', title).replace(' ', '_'), safe='')}"
        entries.append({
            "local_file": local_path.name,
            "title": title,
            "page_id": page.get("pageid"),
            "revision_id": summary.get("revision"),
            "timestamp": summary.get("timestamp"),
            "license_metadata_revision_id": page.get("latest", {}).get("id"),
            "license_metadata_timestamp": page.get("latest", {}).get("timestamp"),
            "source_url": source_url,
            "api_url": f"https://{args.language}.wikipedia.org/w/rest.php/v1/page/{quote(title.replace(' ', '_'), safe='')}/with_html",
            "license": page["license"],
            "license_declaration": args.license_declaration,
            "summary_revision_id": summary.get("revision"),
            "summary_timestamp": summary.get("timestamp"),
            "content_kind": "lead summary extract",
        })
        if args.delay_seconds and index + 1 < len(args.title):
            time.sleep(args.delay_seconds)
    manifest = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "language": args.language,
        "endpoint": "MediaWiki REST page metadata plus page summary extract",
        "license_declaration": args.license_declaration,
        "attribution_note": "Attribute each page using its title and source_url; identify that a summary extract was used.",
        "pages": entries,
    }
    manifest_path = output / "attribution.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fetched {len(entries)} article summaries to {output}; attribution: {manifest_path}")


if __name__ == "__main__":
    main()
