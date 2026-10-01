import glob, re, json, os

def judge_status():
    out = []
    for f in sorted(glob.glob('/tmp/judge/judge_*.log')):
        name = f.split('judge_')[1].replace('.log','')
        try:
            content = open(f).read()
            lines = content.strip().split(chr(10))
            last = lines[-1].strip() if lines else ''
            if last in ('}', ''):
                out.append(name + ': DONE')
            elif 'eval_judge' in last:
                m = re.search(r'(\d+)/(\d+)', last)
                if m:
                    out.append(name + ': ' + m.group(1) + '/' + m.group(2))
                else:
                    out.append(name + ': running')
            else:
                out.append(name + ': ' + last[:60])
        except Exception:
            out.append(name + ': error')
    return chr(10).join(out) if out else '(no judge jobs)'

def judge_accuracy_table():
    L = 'MASim/runs/l_20260408_111046/eval_results'
    rows = []
    # RAG
    for f in sorted(glob.glob(f'{L}/inmem/evaluation_results_rag_*.json')):
        if 'latency' in f: continue
        try:
            d = json.load(open(f))
            s = d.get('summary', {})
            name = os.path.basename(f).replace('evaluation_results_', '').replace('.json', '')
            judged = s.get('judge_enabled', False)
            tag = 'judged' if judged else 'rule-only'
            rows.append(f'rag  {name:30s}  acc={s.get("accuracy",0):.4f}  {s.get("correct",0):4d}/{s.get("total",0)}  [{tag}]')
        except Exception:
            pass
    # ORACLE
    for f in sorted(glob.glob(f'{L}/oracle/evaluation_results_oracle_*.json')):
        if 'latency' in f: continue
        try:
            d = json.load(open(f))
            s = d.get('summary', {})
            name = os.path.basename(f).replace('evaluation_results_', '').replace('.json', '')
            judged = s.get('judge_enabled', False)
            tag = 'judged' if judged else 'rule-only'
            rows.append(f'oracle  {name:28s}  acc={s.get("accuracy",0):.4f}  {s.get("correct",0):4d}/{s.get("total",0)}  [{tag}]')
        except Exception:
            pass
    # VANILLA
    for f in sorted(glob.glob(f'{L}/fc_*.json')):
        if 'latency' in f: continue
        try:
            d = json.load(open(f))
            s = d.get('summary', {})
            name = os.path.basename(f).replace('.json', '')
            rows.append(f'vanilla  {name:27s}  acc={s.get("accuracy",0):.4f}  {s.get("correct",0):4d}/{s.get("total",0)}  [rule-only]')
        except Exception:
            pass
    return chr(10).join(rows) if rows else '(no results)'
