from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
    result = {agent: dict(counter) for agent, counter in grouped.items()}
    result["combined"] = dict(Counter(r.failure_mode for r in records))
    return result

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summarize(records), failure_modes=failure_breakdown(records), examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion="Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.")

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")

    # Generate beautiful HTML report
    html_path = out_dir / "report.html"
    examples_json = json.dumps(report.examples)
    react_failures = report.failure_modes.get("react", {})
    reflexion_failures = report.failure_modes.get("reflexion", {})
    
    react_none = react_failures.get("none", 0)
    react_wrong = react_failures.get("wrong_final_answer", 0)
    react_total = react_none + react_wrong
    react_success_rate = round((react_none / react_total * 100), 2) if react_total > 0 else 0.0
    
    ref_none = reflexion_failures.get("none", 0)
    ref_wrong = reflexion_failures.get("wrong_final_answer", 0)
    ref_total = ref_none + ref_wrong
    ref_success_rate = round((ref_none / ref_total * 100), 2) if ref_total > 0 else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reflexion Agent Benchmark Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --secondary: #a855f7;
            --success: #10b981;
            --danger: #ef4444;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            line-height: 1.6;
            padding: 2rem;
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(168, 85, 247, 0.12) 0px, transparent 50%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 3rem;
            text-align: center;
            animation: fadeIn 0.8s ease-in-out;
        }}

        h1 {{
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .meta-badges {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1.25rem;
            flex-wrap: wrap;
        }}

        .badge {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            color: #d1d5db;
            font-weight: 500;
            backdrop-filter: blur(4px);
        }}

        /* Overview Cards */
        .grid-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
            animation: fadeIn 1s ease-in-out;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
        }}

        .card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: transparent;
        }}

        .card.primary::before {{
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }}

        .card.success::before {{
            background: var(--success);
        }}

        .card.danger::before {{
            background: var(--danger);
        }}

        .card:hover {{
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.25);
            box-shadow: 0 10px 30px var(--primary-glow);
        }}

        .card-title {{
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .card-value {{
            font-size: 2.25rem;
            font-weight: 700;
            display: flex;
            align-items: baseline;
            gap: 0.25rem;
        }}

        .card-subtext {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }}

        /* Metrics Table */
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .section-title::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: var(--card-border);
        }}

        .table-container {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 3rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th, td {{
            padding: 1.25rem 1.5rem;
            border-bottom: 1px solid var(--card-border);
        }}

        th {{
            background: rgba(255, 255, 255, 0.01);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.01);
        }}

        .delta-positive {{
            color: var(--success);
            font-weight: 600;
        }}

        .delta-negative {{
            color: var(--danger);
            font-weight: 600;
        }}

        .agent-name-cell {{
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Chart Section */
        .chart-box {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        @media (max-width: 768px) {{
            .chart-box {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}

        .bar-container {{
            margin-top: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .bar-row {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .bar-label {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 500;
        }}

        .bar-outer {{
            width: 100%;
            height: 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }}

        .bar-inner {{
            height: 100%;
            border-radius: 6px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .bar-inner.success {{
            background: linear-gradient(90deg, #10b981, #34d399);
        }}

        .bar-inner.danger {{
            background: linear-gradient(90deg, #ef4444, #f87171);
        }}

        /* Interactive Examples List */
        .interactive-section {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            margin-bottom: 3rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}

        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            align-items: center;
            justify-content: space-between;
        }}

        .filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .filter-btn {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 0.55rem 1.1rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }}

        .filter-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-color);
        }}

        .filter-btn.active {{
            background: var(--primary);
            border-color: var(--primary);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
        }}

        .search-box {{
            position: relative;
            flex: 1;
            max-width: 400px;
            min-width: 250px;
        }}

        .search-box input {{
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--card-border);
            color: var(--text-color);
            padding: 0.6rem 1rem 0.6rem 2.5rem;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }}

        .search-box input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.15);
        }}

        .search-box svg {{
            position: absolute;
            left: 0.85rem;
            top: 50%;
            transform: translateY(-50%);
            width: 1.1rem;
            height: 1.1rem;
            fill: var(--text-muted);
            pointer-events: none;
        }}

        .example-list {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            max-height: 600px;
            overflow-y: auto;
            padding-right: 0.5rem;
            margin-top: 1rem;
        }}

        .example-list::-webkit-scrollbar {{
            width: 8px;
        }}
        .example-list::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.15);
            border-radius: 4px;
        }}
        .example-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.08);
            border-radius: 4px;
        }}
        .example-list::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.15);
        }}

        .example-item {{
            background: rgba(0, 0, 0, 0.18);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .example-item:hover {{
            border-color: rgba(99, 102, 241, 0.2);
            background: rgba(0, 0, 0, 0.25);
            transform: translateX(4px);
        }}

        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.75rem;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
            padding-bottom: 0.75rem;
        }}

        .example-title {{
            font-weight: 600;
            font-size: 1.05rem;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-badge {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.25rem 0.75rem;
            border-radius: 50px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .status-badge.correct {{
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}

        .status-badge.incorrect {{
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }}

        .example-body {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.75rem;
            font-size: 0.95rem;
        }}

        .field-row {{
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            background: rgba(255, 255, 255, 0.01);
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.02);
        }}

        @media (min-width: 640px) {{
            .field-row {{
                flex-direction: row;
                align-items: flex-start;
                gap: 0.75rem;
            }}
        }}

        .field-label {{
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            min-width: 160px;
        }}

        .field-val {{
            color: #e5e7eb;
            font-weight: 400;
            word-break: break-word;
        }}

        .field-val.gold {{
            color: var(--success);
            font-weight: 500;
        }}

        .field-val.predicted {{
            color: #818cf8;
            font-weight: 500;
        }}

        /* Discussion Section */
        .discussion-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(12px);
            margin-bottom: 3rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }}

        .discussion-text {{
            color: #d1d5db;
            font-size: 1.05rem;
            line-height: 1.8;
            white-space: pre-line;
        }}

        .ext-list {{
            margin-top: 1rem;
            margin-left: 1.5rem;
            color: #d1d5db;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(15px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Reflexion Agent Benchmark</h1>
            <p style="color: var(--text-muted); font-size: 1.1rem; margin-top: 0.25rem;">Báo cáo so sánh ReAct Agent và Reflexion Agent</p>
            <div class="meta-badges">
                <span class="badge">Dataset: {report.meta['dataset']}</span>
                <span class="badge">Chế độ: {report.meta['mode'].upper()}</span>
                <span class="badge">Tổng số bản ghi: {report.meta['num_records']}</span>
                <span class="badge">Mô hình: {report.meta.get('model', 'gemini-3.1-flash-lite')}</span>
            </div>
        </header>

        <!-- Metrics Grid -->
        <div class="grid-cards">
            <div class="card primary">
                <div class="card-title">Độ chính xác ReAct (EM)</div>
                <div class="card-value">{float(react.get('em', 0)) * 100:.1f}%</div>
                <div class="card-subtext">{react.get('count', 0)} câu hỏi thử nghiệm</div>
            </div>
            <div class="card success">
                <div class="card-title">Độ chính xác Reflexion (EM)</div>
                <div class="card-value highlight-success">{float(reflexion.get('em', 0)) * 100:.1f}%</div>
                <div class="card-subtext">{reflexion.get('count', 0)} câu hỏi thử nghiệm</div>
            </div>
            <div class="card primary">
                <div class="card-title">Cải thiện độ chính xác (Delta)</div>
                <div class="card-value highlight-primary">+{float(delta.get('em_abs', 0)) * 100:.1f}%</div>
                <div class="card-subtext">Độ chính xác tăng tuyệt đối</div>
            </div>
            <div class="card">
                <div class="card-title">Số lần thử TB (Reflexion)</div>
                <div class="card-value">{reflexion.get('avg_attempts', 0)}</div>
                <div class="card-subtext">Số lần thử tối đa cấu hình: 3</div>
            </div>
        </div>

        <!-- Detail Table -->
        <div class="section-title">Bảng so sánh chi tiết</div>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>ReAct Agent</th>
                        <th>Reflexion Agent</th>
                        <th>Delta</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="agent-name-cell">Exact Match (EM) Accuracy</td>
                        <td>{float(react.get('em', 0)) * 100:.1f}%</td>
                        <td>{float(reflexion.get('em', 0)) * 100:.1f}%</td>
                        <td class="delta-positive">+{float(delta.get('em_abs', 0)) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td class="agent-name-cell">Số lần thử trung bình (Avg Attempts)</td>
                        <td>{react.get('avg_attempts', 0)}</td>
                        <td>{reflexion.get('avg_attempts', 0)}</td>
                        <td class="delta-positive">+{delta.get('attempts_abs', 0)}</td>
                    </tr>
                    <tr>
                        <td class="agent-name-cell">Số Tokens trung bình (Avg Tokens)</td>
                        <td>{react.get('avg_token_estimate', 0)}</td>
                        <td>{reflexion.get('avg_token_estimate', 0)}</td>
                        <td class="delta-negative">+{delta.get('tokens_abs', 0)}</td>
                    </tr>
                    <tr>
                        <td class="agent-name-cell">Độ trễ trung bình (Avg Latency)</td>
                        <td>{react.get('avg_latency_ms', 0):.0f} ms</td>
                        <td>{reflexion.get('avg_latency_ms', 0):.0f} ms</td>
                        <td class="delta-positive">{delta.get('latency_abs', 0):.0f} ms</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Failure Modes Charts -->
        <div class="section-title">Phân tích lỗi sai (Failure Modes)</div>
        <div class="chart-box">
            <div class="chart-card">
                <h3 style="font-weight: 600; margin-bottom: 0.5rem; font-size: 1.1rem;">Tỷ lệ lỗi - ReAct</h3>
                <div class="bar-container">
                    <div class="bar-row">
                        <div class="bar-label">
                            <span>Thành công (None)</span>
                            <span>{react_none} ({react_success_rate}%)</span>
                        </div>
                        <div class="bar-outer">
                            <div class="bar-inner success" style="width: {react_success_rate}%"></div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">
                            <span>Lỗi trả lời sai (Wrong Final Answer)</span>
                            <span>{react_wrong} ({100 - react_success_rate:.2f}%)</span>
                        </div>
                        <div class="bar-outer">
                            <div class="bar-inner danger" style="width: {100 - react_success_rate}%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="chart-card">
                <h3 style="font-weight: 600; margin-bottom: 0.5rem; font-size: 1.1rem;">Tỷ lệ lỗi - Reflexion</h3>
                <div class="bar-container">
                    <div class="bar-row">
                        <div class="bar-label">
                            <span>Thành công (None)</span>
                            <span>{ref_none} ({ref_success_rate}%)</span>
                        </div>
                        <div class="bar-outer">
                            <div class="bar-inner success" style="width: {ref_success_rate}%"></div>
                        </div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">
                            <span>Lỗi trả lời sai (Wrong Final Answer)</span>
                            <span>{ref_wrong} ({100 - ref_success_rate:.2f}%)</span>
                        </div>
                        <div class="bar-outer">
                            <div class="bar-inner danger" style="width: {100 - ref_success_rate}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Discussion Card -->
        <div class="section-title">Thảo luận & Đánh giá</div>
        <div class="discussion-card">
            <div class="discussion-text">{report.discussion}</div>
            <h4 style="margin-top: 1.5rem; font-weight: 600; color: #ffffff;">Các tính năng mở rộng đã triển khai:</h4>
            <ul class="ext-list">
                {"".join(f"<li>{item}</li>" for item in report.extensions)}
            </ul>
        </div>

        <!-- Interactive Examples -->
        <div class="section-title">Danh sách chi tiết các ví dụ chạy thực tế</div>
        <div class="interactive-section">
            <div class="controls">
                <div class="filters" id="filtersContainer">
                    <button class="filter-btn active" onclick="filterType('all')">Tất cả ({len(report.examples)})</button>
                    <button class="filter-btn" onclick="filterType('react')">ReAct ({react.get('count', 0)})</button>
                    <button class="filter-btn" onclick="filterType('reflexion')">Reflexion ({reflexion.get('count', 0)})</button>
                    <button class="filter-btn" onclick="filterStatus('correct')">Đúng</button>
                    <button class="filter-btn" onclick="filterStatus('incorrect')">Sai</button>
                </div>
                <div class="search-box">
                    <svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
                    <input type="text" id="searchInput" placeholder="Tìm kiếm câu hỏi hoặc câu trả lời..." onkeyup="searchExamples()">
                </div>
            </div>
            
            <div class="example-list" id="examplesContainer">
                <!-- Javascript will render examples here -->
            </div>
        </div>
    </div>

    <script>
        const rawExamples = {examples_json};
        let activeTypeFilter = 'all';
        let activeStatusFilter = 'all';
        let searchQuery = '';

        function renderExamples() {{
            const container = document.getElementById('examplesContainer');
            container.innerHTML = '';

            const filtered = rawExamples.filter(item => {{
                // Type Filter
                if (activeTypeFilter !== 'all' && item.agent_type !== activeTypeFilter) return false;
                
                // Status Filter
                if (activeStatusFilter === 'correct' && !item.is_correct) return false;
                if (activeStatusFilter === 'incorrect' && item.is_correct) return false;

                // Search Filter
                if (searchQuery) {{
                    const q = searchQuery.toLowerCase();
                    const goldMatch = item.gold_answer.toLowerCase().includes(q);
                    const predMatch = item.predicted_answer.toLowerCase().includes(q);
                    return goldMatch || predMatch;
                }}
                
                return true;
            }});

            if (filtered.length === 0) {{
                container.innerHTML = '<div style="text-align: center; padding: 2rem; color: var(--text-muted);">Không tìm thấy bản ghi nào khớp với bộ lọc.</div>';
                return;
            }}

            filtered.forEach(item => {{
                const card = document.createElement('div');
                card.className = 'example-item';
                
                const statusClass = item.is_correct ? 'correct' : 'incorrect';
                const statusText = item.is_correct ? 'Chính xác' : 'Lỗi';

                card.innerHTML = `
                    <div class="example-header">
                        <span class="example-title">ID: ${{item.qid}}</span>
                        <span class="status-badge ${{statusClass}}">${{statusText}}</span>
                    </div>
                    <div class="example-body">
                        <div class="field-row">
                            <span class="field-label">Phương thức Agent:</span>
                            <span class="field-val" style="text-transform: uppercase; font-weight: 600;">${{item.agent_type}}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-label">Đáp án chuẩn (Gold):</span>
                            <span class="field-val gold">${{item.gold_answer}}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-label">Đáp án dự đoán:</span>
                            <span class="field-val predicted">${{item.predicted_answer}}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-label">Số lần thử:</span>
                            <span class="field-val">${{item.attempts}}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-label">Kiểu lỗi (Failure):</span>
                            <span class="field-val" style="color: ${{item.is_correct ? 'var(--text-muted)' : 'var(--danger)'}}">${{item.failure_mode}}</span>
                        </div>
                        <div class="field-row">
                            <span class="field-label">Số bản phản chiếu:</span>
                            <span class="field-val">${{item.reflection_count}}</span>
                        </div>
                    </div>
                `;
                container.appendChild(card);
            }});
        }}

        function filterType(type) {{
            activeTypeFilter = type;
            activeStatusFilter = 'all';
            updateButtons();
            renderExamples();
        }}

        function filterStatus(status) {{
            activeStatusFilter = status;
            activeTypeFilter = 'all';
            updateButtons();
            renderExamples();
        }}

        function updateButtons() {{
            const btns = document.querySelectorAll('.filter-btn');
            btns.forEach(btn => btn.classList.remove('active'));
            
            if (activeTypeFilter === 'all' && activeStatusFilter === 'all') {{
                btns[0].classList.add('active');
            }} else if (activeTypeFilter === 'react') {{
                btns[1].classList.add('active');
            }} else if (activeTypeFilter === 'reflexion') {{
                btns[2].classList.add('active');
            }} else if (activeStatusFilter === 'correct') {{
                btns[3].classList.add('active');
            }} else if (activeStatusFilter === 'incorrect') {{
                btns[4].classList.add('active');
            }}
        }}

        function searchExamples() {{
            searchQuery = document.getElementById('searchInput').value;
            renderExamples();
        }}

        // Initial render
        renderExamples();
    </script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
    return json_path, md_path
