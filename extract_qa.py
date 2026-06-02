import csv
import re
import os

input_file = '/home/efison-kristo/rag-for-l1-aleleon-hpc-support/output/podman_output_rag_Qwen3.5-35B-A3B-GPTQ-Int4_#12'
output_file = '/home/efison-kristo/rag-for-l1-aleleon-hpc-support/output/podman_output_rag_Qwen3.5-35B-A3B-GPTQ-Int4_#12.csv'

questions_answers = []

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_question = None
current_answer_lines = []
state = 0 # 0: looking for question, 1: skipping debug, 2: collecting answer, 3: skipping sources

for line in lines:
    if line.startswith('============================================================'):
        if current_question:
            answer = ''.join(current_answer_lines).strip()
            questions_answers.append({
                'pertanyaan': current_question,
                'jawaban': answer
            })
        current_question = None
        current_answer_lines = []
        state = 0
        continue
    
    if state == 0:
        if line.startswith('[Q'):
            m = re.match(r'\[Q\d+/\d+\]\s*(.*)', line)
            if m:
                current_question = m.group(1).strip()
            else:
                current_question = line.strip()
            state = 1
    elif state == 1:
        if line.strip() == '' or line.startswith('----'):
            continue
        if line.startswith('    '):
            continue
        current_answer_lines.append(line)
        state = 2
    elif state == 2:
        if '📚 Sumber' in line:
            state = 3
        else:
            current_answer_lines.append(line)

if current_question:
    answer = ''.join(current_answer_lines).strip()
    questions_answers.append({
        'pertanyaan': current_question,
        'jawaban': answer
    })

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['pertanyaan', 'jawaban'])
    for qa in questions_answers:
        writer.writerow([qa['pertanyaan'], qa['jawaban']])

print(f"Extracted {len(questions_answers)} Q&A pairs to {output_file}")
