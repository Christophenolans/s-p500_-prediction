#!/usr/bin/env python3
"""Annual walk-forward backtest for data/results 1 horizon 1/ files."""
import argparse, csv, math, os, json, urllib.request, urllib.error
from urllib.parse import quote
from collections import defaultdict
from datetime import date

BIN_SIZE = 0.1
DEFAULT_COST_BPS = 5.0
DEFAULT_MIN_BIN_COUNT = 20
GITHUB_API_TREE = "https://api.github.com/repos/Christophenolans/s-p500_-prediction/git/trees/guide?recursive=1"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Christophenolans/s-p500_-prediction/guide/"
REPO_HORIZON_FOLDER = "data/results 1 horizon 1/"

def download_horizon1_files(destination):
    """Download every tracked file under the Horizon-1 results folder from branch guide."""
    req = urllib.request.Request(GITHUB_API_TREE, headers={"User-Agent": "SP500-Horizon1-Backtest"})
    with urllib.request.urlopen(req, timeout=30) as response:
        tree = json.loads(response.read().decode("utf-8"))
    files = [item["path"] for item in tree.get("tree", [])
             if item.get("type") == "blob" and item["path"].startswith(REPO_HORIZON_FOLDER)]
    if not files:
        raise RuntimeError(f"No files found under {REPO_HORIZON_FOLDER!r} on branch guide.")
    downloaded = 0
    for repo_path in files:
        relative = repo_path[len(REPO_HORIZON_FOLDER):]
        local_path = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = GITHUB_RAW_BASE + quote(repo_path, safe="/")
        req = urllib.request.Request(url, headers={"User-Agent": "SP500-Horizon1-Backtest"})
        with urllib.request.urlopen(req, timeout=30) as response, open(local_path, "wb") as output:
            output.write(response.read())
        downloaded += 1
    print(f"Downloaded {downloaded} Horizon-1 files from GitHub branch guide into: {destination}")
    return downloaded

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def number(v):
    try:
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null", "na", "n/a", "."}: return None
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None

def dt(v):
    try: return date.fromisoformat(str(v).strip()[:10])
    except (ValueError, TypeError): return None

def ret_pct(row):
    op, fop = number(row.get("Open")), number(row.get("Future_Open"))
    d, fd = dt(row.get("Date")), dt(row.get("Future_Date"))
    if op is None or fop is None or op <= 0 or not d or not fd or fd <= d: return None
    # Recompute return from prices, not the ambiguously named Return_14_Trading_Days_Percent column.
    return (fop / op - 1.0) * 100.0

def bin_score(score): return round(math.floor(score / BIN_SIZE) * BIN_SIZE, 10)

def calibrate(rows):
    bins = defaultdict(list)
    for row in rows:
        s, r = number(row.get("Score")), ret_pct(row)
        if s is not None and r is not None: bins[bin_score(s)].append(r)
    return {b: {"n":len(rs), "mean":sum(rs)/len(rs), "up_pct":100*sum(x>0 for x in rs)/len(rs)} for b,rs in bins.items()}

def max_drawdown(equity):
    peak, worst = 1.0, 0.0
    for x in equity:
        peak = max(peak,x)
        if peak > 0: worst = min(worst,x/peak-1)
    return 100*worst

def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def simulate(train_rows,test_rows,window,year,min_count,cost_bps):
    cal=calibrate(train_rows); daily=[]; position=0; equity=1.0; bh=1.0; changes=0
    cost_pct=max(0.0,cost_bps)/100.0 # basis points to percentage points
    for row in sorted(test_rows,key=lambda x:dt(x.get("Date")) or date.min):
        d,s,r=dt(row.get("Date")),number(row.get("Score")),ret_pct(row)
        if not d or d.year!=year or s is None or r is None: continue
        b=bin_score(s); c=cal.get(b)
        expected=c["mean"] if c and c["n"]>=min_count else None
        up=c["up_pct"] if c and c["n"]>=min_count else None
        desired=1 if expected is not None and expected>0 else 0
        signal="BUY/HOLD" if desired else ("SELL/CASH" if expected is not None else "NO_CALIBRATION")
        changed=desired!=position; fee=cost_pct if changed else 0.0; changes+=int(changed)
        strategy_ret=desired*r-fee
        equity*=1+strategy_ret/100; bh*=1+r/100; position=desired
        daily.append({"train_window":window,"test_year":year,"date":d.isoformat(),"score":s,
          "score_bin_lower":b,"train_bin_count":c["n"] if c else 0,
          "train_probability_up_pct":"" if up is None else up,
          "train_expected_return_pct":"" if expected is None else expected,
          "signal":signal,"position":desired,"realized_horizon1_return_pct":r,
          "transaction_cost_pct":fee,"strategy_return_pct":strategy_ret,
          "strategy_equity":equity,"buy_hold_equity":bh})
    if not daily:
        summary={"train_window":window,"test_year":year,"rows_tested":0,"position_changes":0,
          "strategy_total_return_pct":"","buy_hold_total_return_pct":"","strategy_max_drawdown_pct":"",
          "buy_hold_max_drawdown_pct":"","positive_strategy_days_pct":"","note":"No valid rows/scores"}
    else:
        sr=[float(x["strategy_return_pct"]) for x in daily]
        summary={"train_window":window,"test_year":year,"rows_tested":len(daily),"position_changes":changes,
          "strategy_total_return_pct":(equity-1)*100,"buy_hold_total_return_pct":(bh-1)*100,
          "strategy_max_drawdown_pct":max_drawdown([float(x["strategy_equity"]) for x in daily]),
          "buy_hold_max_drawdown_pct":max_drawdown([float(x["buy_hold_equity"]) for x in daily]),
          "positive_strategy_days_pct":100*sum(x>0 for x in sr)/len(sr),
          "note":"Long-only illustrative simulation; not a forecast."}
    return daily,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default=os.path.join("data","results 1 horizon 1"))
    p.add_argument("--no-download", action="store_true", help="Use local Horizon-1 files without syncing from GitHub")
    p.add_argument("--out-dir",default=os.path.join("result","horizon1_trade_backtest"))
    p.add_argument("--min-bin-count",type=int,default=DEFAULT_MIN_BIN_COUNT)
    p.add_argument("--cost-bps",type=float,default=DEFAULT_COST_BPS)
    p.add_argument("--first-train-start",type=int,default=2000)
    p.add_argument("--last-train-end",type=int,default=2025)
    a=p.parse_args(); all_daily=[]; summaries=[]; skipped=[]
    if not a.no_download:
        try:
            download_horizon1_files(a.data_dir)
        except Exception as exc:
            # Continue with local files if available; fail clearly if no local data exists.
            if not os.path.isdir(a.data_dir) or not any(
                os.path.isfile(os.path.join(root, "spy_daily_score.csv"))
                for root, _, _ in os.walk(a.data_dir)
            ):
                raise SystemExit(f"Could not download Horizon-1 files and no usable local data was found: {exc}")
            print(f"WARNING: GitHub sync failed; using existing local files. Details: {exc}")
    # Train YYYY-(YYYY+6), test following year using rows from the next overlapping window.
    for start in range(a.first_train_start,a.last_train_end-5):
        end,year=start+6,start+7
        train_file=os.path.join(a.data_dir,f"{start}-{end}","spy_daily_score.csv")
        test_file=os.path.join(a.data_dir,f"{start+1}-{end+1}","spy_daily_score.csv")
        if not os.path.isfile(train_file):
            skipped.append({"train_window":f"{start}-{end}","test_year":year,"reason":f"Missing {train_file}"}); continue
        if not os.path.isfile(test_file):
            skipped.append({"train_window":f"{start}-{end}","test_year":year,"reason":f"Missing {test_file}"}); continue
        train_rows=read_csv(train_file)
        test_rows=[r for r in read_csv(test_file) if (dt(r.get("Date")) or date.min).year==year]
        daily,summary=simulate(train_rows,test_rows,f"{start}-{end}",year,max(1,a.min_bin_count),max(0,a.cost_bps))
        all_daily.extend(daily); summaries.append(summary)
        print(f"Train {start}-{end} -> Test {year}: {summary['rows_tested']} rows; {summary['position_changes']} position changes")
    out=a.out_dir
    write_csv(os.path.join(out,"daily_backtest.csv"),all_daily,["train_window","test_year","date","score","score_bin_lower","train_bin_count","train_probability_up_pct","train_expected_return_pct","signal","position","realized_horizon1_return_pct","transaction_cost_pct","strategy_return_pct","strategy_equity","buy_hold_equity"])
    write_csv(os.path.join(out,"annual_summary.csv"),summaries,["train_window","test_year","rows_tested","position_changes","strategy_total_return_pct","buy_hold_total_return_pct","strategy_max_drawdown_pct","buy_hold_max_drawdown_pct","positive_strategy_days_pct","note"])
    write_csv(os.path.join(out,"skipped_windows.csv"),skipped,["train_window","test_year","reason"])
    print(f"Saved CSV outputs to {out}; skipped windows: {len(skipped)}")

if __name__=="__main__": main()
