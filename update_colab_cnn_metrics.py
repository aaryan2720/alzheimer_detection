import json

with open('GNN_Alzheimer_GAT_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_text = ''.join(cell['source'])
        if '0.3953' in source_text or 'CNN Baseline' in source_text or 'CNN_METRICS' in source_text:
            print(f"Modifying cell {i}")
            # Replace old values with true ones
            new_source = []
            for line in cell['source']:
                line = line.replace('0.3953', '0.9373')
                line = line.replace('0.2458', '0.9379')
                line = line.replace('0.2460', '0.9385')
                line = line.replace('0.2459', '0.9373')
                line = line.replace('0.4902', '0.9852')
                new_source.append(line)
            cell['source'] = new_source
            updated = True

if updated:
    with open('GNN_Alzheimer_GAT_Colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("GAT Colab notebook updated successfully!")
else:
    print("Target metrics not found in GAT Colab notebook.")
