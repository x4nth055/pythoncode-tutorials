"""
Track Your AI Visibility with Python & RankBits API.

This script demonstrates the full workflow:
1. Check your RankBits account and plan
2. Create an AI visibility scan for any domain
3. Poll until the scan completes
4. Parse the results and generate visualizations

Requirements:
    pip install requests matplotlib

Usage:
    export RANKBITS_TOKEN="rb_your_token_here"
    python ai_visibility_tracker.py
"""

import os
import sys
import time
import json
from datetime import datetime

import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOKEN = os.environ.get("RANKBITS_TOKEN", "rb_your_token_here")
BASE_URL = "https://rankbits.com/v1"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

# The domain you want to scan
TARGET_URL = "https://thepythoncode.com"

# Free engines to use (omit "paid" providers like openai_pro, claude_pro, gemini_pro)
ENGINES = ["openai", "gemini", "perplexity", "claude", "google_ai_mode"]

# Number of AI-generated prompts (plan caps apply)
PROMPT_COUNT = 5


# ---------------------------------------------------------------------------
# Helper: pretty-print JSON
# ---------------------------------------------------------------------------

def print_json(obj: dict, title: str = "") -> None:
    """Print a dictionary as formatted JSON."""
    if title:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# Step 1 – Check your account
# ---------------------------------------------------------------------------

def check_account() -> dict:
    """Fetch plan info and credit usage from /v1/me."""
    resp = requests.get(f"{BASE_URL}/me", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    plan = data["plan"]
    resp_info = plan["responses"]

    print("🔑  Account")
    print(f"   Plan:     {plan['label']} (${plan['price_usd']}/mo)")
    print(f"   Monthly:  {resp_info['used']}/{resp_info['monthly_limit']} responses")
    print(f"   Credits:  {resp_info['purchased_remaining']} purchased remaining")
    print(f"   Engines:  {len(plan['allowed_provider_keys'])} available")
    return data


# ---------------------------------------------------------------------------
# Step 2 – Create a scan
# ---------------------------------------------------------------------------

def create_scan(
    url: str,
    prompt_count: int = 5,
    providers: list[str] | None = None,
) -> dict:
    """Submit an async scan and return the public ID."""
    payload: dict = {"url": url, "prompt_count": prompt_count}
    if providers:
        payload["providers"] = providers

    resp = requests.post(f"{BASE_URL}/scans", headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()

    scan = data["scan"]
    print(f"\n🚀  Scan created")
    print(f"   ID:        {scan['public_id']}")
    print(f"   Domain:    {scan['domain']}")
    print(f"   Status:    {scan['status']}")
    print(f"   View live: https://rankbits.com{data['links']['app']}")
    return data


# ---------------------------------------------------------------------------
# Step 3 – Poll until done
# ---------------------------------------------------------------------------

def poll_scan(public_id: str, poll_seconds: float = 3.0, max_wait: float = 300.0) -> dict:
    """Poll /v1/scans/{id} until status is 'done' or timeout."""
    url = f"{BASE_URL}/scans/{public_id}"
    start = time.time()
    last_completed = 0

    print(f"\n⏳  Polling scan {public_id} ...")
    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            raise TimeoutError(f"Scan did not complete within {max_wait}s")

        resp = requests.get(url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        status = data["scan"]["status"]
        progress = data.get("progress", {})
        completed = progress.get("completed_results", 0)
        expected = progress.get("expected_results", 0)

        # Print progress when it changes
        if completed != last_completed:
            pct = (completed / expected * 100) if expected else 0
            print(f"   [{status}] {completed}/{expected} ({pct:.0f}%)")
            last_completed = completed

        if status == "done":
            print("   ✅  Scan complete!")
            return data
        if status in ("error", "failed"):
            raise RuntimeError(f"Scan failed: {data}")

        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Step 4 – Parse & display results
# ---------------------------------------------------------------------------

def summarize_results(data: dict) -> None:
    """Print a human-readable summary of scan results."""
    aggregate = data.get("aggregate", {})
    overall = aggregate.get("overall", {})
    providers = aggregate.get("providers", {})
    results = data.get("results", [])
    prompts = data.get("prompts", [])

    # ---- 4a. Overview ----
    print(f"\n📊  Visibility Summary for {data['scan']['domain']}")
    print(f"   Overall score:      {overall.get('score', 'N/A')}")
    print(f"   Mention rate:       {overall.get('mention_rate', 0):.1f}%")
    print(f"   Citation rate:      {overall.get('citation_rate', 0):.1f}%")
    print(f"   Total results:      {len(results)} rows")

    # ---- 4b. Per-engine breakdown ----
    print(f"\n🤖  Engine Breakdown")
    print(f"   {'Engine':<20s} {'Score':>7s} {'Mention%':>9s} {'Citation%':>10s}")
    print(f"   {'-'*46}")
    for key, pdata in sorted(providers.items(), key=lambda x: -x[1].get("score", 0)):
        print(
            f"   {key:<20s} {pdata.get('score', 0):>7.1f} "
            f"{pdata.get('mention_rate', 0):>8.1f}% {pdata.get('citation_rate', 0):>9.1f}%"
        )

    # ---- 4c. Prompts used ----
    print(f"\n💬  Prompts ({len(prompts)})")
    for p in prompts:
        print(f"   • {p['text']}")

    # ---- 4d. Share of voice (top 5) ----
    sov = aggregate.get("share_of_voice", [])
    if sov:
        print(f"\n🔗  Top Cited Domains (Share of Voice)")
        for entry in sov[:5]:
            print(f"   {entry['domain']:40s} {entry.get('citation_count', 0)} citations")

    # ---- 4e. Where we were found ----
    found = [r for r in results if r.get("brand_mentioned") or r.get("brand_cited")]
    if found:
        print(f"\n✅  Where {data['scan']['domain']} Appeared ({len(found)}/{len(results)})")
        for r in found:
            mentioned = "✅" if r["brand_mentioned"] else "❌"
            cited = "✅" if r["brand_cited"] else "❌"
            print(f"   [{r['provider']:20s}] Mentioned: {mentioned}  Cited: {cited}")
            print(f"   Prompt: {r['prompt'][:100]}")
    else:
        print(f"\n⚠️   {data['scan']['domain']} was NOT mentioned or cited in any result!")
        print("   Time to improve your AI visibility! → https://rankbits.com")


# ---------------------------------------------------------------------------
# Step 5 – Generate charts
# ---------------------------------------------------------------------------

def generate_charts(data: dict, output_dir: str = ".") -> None:
    """Create matplotlib charts from scan results."""
    aggregate = data.get("aggregate", {})
    providers = aggregate.get("providers", {})
    domain = data["scan"]["domain"]

    if not providers:
        print("⚠️  No provider data to chart.")
        return

    # Sort engines by score descending
    engines = sorted(providers.items(), key=lambda x: -x[1].get("score", 0))
    names = [e[0].replace("_", " ").title() for e in engines]
    scores = [e[1].get("score", 0) for e in engines]
    mention_rates = [e[1].get("mention_rate", 0) for e in engines]
    citation_rates = [e[1].get("citation_rate", 0) for e in engines]

    # Colors
    bar_color = "#7c3aed"
    mention_color = "#10b981"
    citation_color = "#f59e0b"

    # ---- Chart 1: Scores by engine ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.barh(names, scores, color=bar_color, edgecolor="white", linewidth=0.5, height=0.5)
    ax1.set_xlabel("Visibility Score (0–100)", fontsize=11)
    ax1.set_title(f"AI Visibility Score by Engine — {domain}", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    for bar, val in zip(bars, scores):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=10, fontweight="semibold")
    ax1.set_xlim(0, max(scores) * 1.3 + 5 if max(scores) > 0 else 30)
    plt.tight_layout()
    fig1.savefig(f"{output_dir}/engine_scores.png", dpi=150)
    print(f"\n📈  Chart saved: {output_dir}/engine_scores.png")

    # ---- Chart 2: Mention vs Citation rates ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], mention_rates, width, label="Mention Rate %",
            color=mention_color, edgecolor="white", linewidth=0.5)
    ax2.bar([i + width / 2 for i in x], citation_rates, width, label="Citation Rate %",
            color=citation_color, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title(f"Mention vs Citation Rate — {domain}", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_ylim(0, max(max(mention_rates), max(citation_rates)) * 1.4 + 5)
    plt.tight_layout()
    fig2.savefig(f"{output_dir}/mention_vs_citation.png", dpi=150)
    print(f"📈  Chart saved: {output_dir}/mention_vs_citation.png")

    # ---- Chart 3: Results grid (heatmap-style table) ----
    results = data.get("results", [])
    if results:
        # Build a matrix: rows=prompts, cols=engines
        prompt_texts = sorted({r["prompt"][:60] for r in results})
        engine_names = sorted({r["provider"] for r in results})

        matrix = []
        for pt in prompt_texts:
            row = []
            for eng in engine_names:
                match = [r for r in results if r["prompt"].startswith(pt[:30]) and r["provider"] == eng]
                if match:
                    m = match[0]
                    if m["brand_cited"]:
                        row.append(2)  # cited (best)
                    elif m["brand_mentioned"]:
                        row.append(1)  # mentioned
                    else:
                        row.append(0)  # absent
                else:
                    row.append(0)
            matrix.append(row)

        fig3, ax3 = plt.subplots(figsize=(max(8, len(engine_names) * 1.2),
                                           max(5, len(prompt_texts) * 0.6)))
        cmap = plt.cm.RdYlGn
        im = ax3.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)

        ax3.set_xticks(range(len(engine_names)))
        ax3.set_xticklabels([e.replace("_", " ").title() for e in engine_names],
                            rotation=30, ha="right", fontsize=9)
        ax3.set_yticks(range(len(prompt_texts)))
        ax3.set_yticklabels(prompt_texts, fontsize=8)

        # Add text in each cell
        for i in range(len(prompt_texts)):
            for j in range(len(engine_names)):
                val = matrix[i][j]
                symbol = {0: "○", 1: "▲", 2: "★"}[val]
                ax3.text(j, i, symbol, ha="center", va="center",
                         fontsize=14, color="black" if val == 2 else "white")

        ax3.set_title(f"Presence Grid — {domain}\n○ Absent  ▲ Mentioned  ★ Cited",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig3.savefig(f"{output_dir}/presence_grid.png", dpi=150)
        print(f"📈  Chart saved: {output_dir}/presence_grid.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if TOKEN == "rb_your_token_here":
        print("❌  Set your RANKBITS_TOKEN environment variable first.")
        print("    Get one at: https://rankbits.com/signup")
        sys.exit(1)

    print(f"🎯  Tracking AI visibility for: {TARGET_URL}")
    print(f"    Engines: {', '.join(ENGINES)}")

    # 1. Check account
    check_account()

    # 2. Start scan
    scan_data = create_scan(TARGET_URL, prompt_count=PROMPT_COUNT, providers=ENGINES)
    public_id = scan_data["scan"]["public_id"]

    # 3. Poll until complete
    results = poll_scan(public_id)

    # 4. Summarize
    summarize_results(results)

    # 5. Charts
    generate_charts(results)

    print("\n✨  Done! Track ongoing visibility at https://rankbits.com")


if __name__ == "__main__":
    main()
