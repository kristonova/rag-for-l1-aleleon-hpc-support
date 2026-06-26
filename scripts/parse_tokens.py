#!/usr/bin/env python3
import re, sys

log_file = sys.argv[1] if len(sys.argv) > 1 else '/home/efison-kristo/rag-for-l1-aleleon-hpc-support/output/podman_output_rag_Qwen3.5-35B-A3B-GPTQ-Int4_#13.txt'

with open(log_file, 'r') as f:
    content = f.read()

q_pattern = re.compile(r'\[Q(\d+)/(\d+)\]\s+(.*)')
token_pattern = re.compile(r'\[(\w+)\] Tokens .+ input: (\d+), output: (\d+), total: (\d+)')

questions = []
current_q = None

for line in content.split('\n'):
    line = line.strip()
    q_match = q_pattern.search(line)
    if q_match:
        if current_q:
            questions.append(current_q)
        current_q = {'num': int(q_match.group(1)), 'question': q_match.group(3), 'calls': {}}
    if current_q:
        t_match = token_pattern.search(line)
        if t_match:
            current_q['calls'][t_match.group(1)] = {
                'input': int(t_match.group(2)),
                'output': int(t_match.group(3))
            }
if current_q:
    questions.append(current_q)

print(f'| Q# | Pertanyaan | is_relevant (in/out) | generate_response (in/out) | source_justifications (in/out) | Total Input | Total Output | Total |')
print(f'|---|---|---|---|---|---:|---:|---:|')

gi, go = 0, 0
for q in questions:
    n = q['num']
    qt = q['question'][:55] + ('...' if len(q['question']) > 55 else '')
    r = q['calls'].get('is_question_relevant', {})
    g = q['calls'].get('generate_response', {})
    s = q['calls'].get('generate_source_justifications', {})
    rs = f"{r.get('input','-')}/{r.get('output','-')}" if r else '-'
    gs = f"{g.get('input','-')}/{g.get('output','-')}" if g else '-'
    ss = f"{s.get('input','-')}/{s.get('output','-')}" if s else '-'
    ti = sum(c.get('input', 0) for c in q['calls'].values())
    to_ = sum(c.get('output', 0) for c in q['calls'].values())
    gi += ti
    go += to_
    print(f'| Q{n} | {qt} | {rs} | {gs} | {ss} | {ti:,} | {to_:,} | {ti+to_:,} |')

print(f'|---|---|---|---|---|---:|---:|---:|')
print(f'| | **GRAND TOTAL** | | | | **{gi:,}** | **{go:,}** | **{gi+go:,}** |')
print(f'| | **Rata-rata/Q** | | | | **{gi//len(questions):,}** | **{go//len(questions):,}** | **{(gi+go)//len(questions):,}** |')
print(f'\nTotal: {len(questions)} pertanyaan')
