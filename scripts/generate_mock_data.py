"""Generate compliant synthetic benchmark data for SemEval-2027 Task 3 local smoke testing."""
import json
from pathlib import Path
import random

def generate_mock_benchmark(output_root: Path):
    output_root = Path(output_root)
    languages = ['EN', 'IT']
    periods = ['1880-1899', '1900-1919', '1920-1939', '1940-1959']
    words_per_lang = {
        'EN': ['apple', 'record', 'plane'],
        'IT': ['banco', 'piano', 'calcio'],
    }

    for lang, words in words_per_lang.items():
        lang_dir = output_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        st1_path = lang_dir / 'subtask1.jsonl'
        st2_path = lang_dir / 'subtask2.jsonl'

        with open(st1_path, 'w', encoding='utf-8') as f1, open(st2_path, 'w', encoding='utf-8') as f2:
            sent_counter = 0
            for word in words:
                for period in periods:
                    for i in range(15):
                        sent_counter += 1
                        sentence_id = f'sent_{sent_counter}'
                        period_idx = periods.index(period)
                        if period_idx == 0:
                            senses = [0] if random.random() < 0.8 else [1]
                        elif period_idx == 1:
                            senses = [0] if random.random() < 0.5 else [1]
                        else:
                            senses = [1] if random.random() < 0.7 else [2]

                        st1_record = {
                            'word': word,
                            'period_label': period,
                            'sentence_id': sentence_id,
                            'sentence': f'Historical sentence {sentence_id} containing target word {word}.',
                            'label': senses,
                        }
                        f1.write(json.dumps(st1_record, ensure_ascii=False) + chr(10))

                        st2_record = {
                            'word': word,
                            'sentence_id': sentence_id,
                            'sentence': f'Historical sentence {sentence_id} containing target word {word}.',
                            'target_sense_definition': f'The target meaning definition for {word}.',
                            'label': 1 if 1 in senses else 0,
                        }
                        f2.write(json.dumps(st2_record, ensure_ascii=False) + chr(10))

    print(f'Generated mock gold data in: {output_root}')

if __name__ == '__main__':
    generate_mock_benchmark(Path('data/mock'))
