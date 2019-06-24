import subprocess
import os
import tqdm


def shell(command):
    return subprocess.run(command, check=True, shell=True)


def experiment(train, test_first, test_joined):
    shell(f'python qanta/main.py ir train --jeopardy --asr-data data/{train} --no-clean --asr')
    shell(f'python qanta/main.py ir asr-eval --skip-title-check --eval-data data/{test_first}')
    shell(f'python qanta/main.py ir asr-eval --skip-title-check --eval-data data/{test_joined}')


def main():
    train_test_files = [
        (
            'asr_qanta.train.2018.04.18.json',
            'asr_qanta.test.first.2018.04.18.json',
            'asr_qanta.test.joined.2018.04.18.json'
        ),
        (
            'asr_qanta.train.nounk.2018.04.18.json',
            'asr_qanta.test.nounk.first.2018.04.18.json',
            'asr_qanta.test.nounk.joined.2018.04.18.json'
        ),
        (
            'asr_qanta.train.expandedv1.2018.04.18.json',
            'asr_qanta.test.expandedv1.first.2018.04.18.json',
            'asr_qanta.test.expandedv1.joined.2018.04.18.json'
        ),
        (
            'asr_qanta.train.expandedv2.2018.04.18.json',
            'asr_qanta.test.expandedv2.first.2018.04.18.json',
            'asr_qanta.test.expandedv2.joined.2018.04.18.json'
        ),
        ('asr_qanta.train.2018.04.18.json', 'human_all.first.json', 'human_all.joined.json'),
        ('jeopardy.train.json', 'jeopardy.test.json', 'jeopardy.test.json'),
        ('jeopardy.nounk.train.json', 'jeopardy.nounk.test.json', 'jeopardy.nounk.test.json'),
        ('searchqa.train.json', 'searchqa.test.json', 'searchqa.test.json'),
        ('searchqa.train.json', 'humanjeopardy.json', 'humanjeopardy.json'),
        ('searchqa.train.json', 'jeopardy.humanclean-formmatted.json', 'jeopardy.humanclean-formmatted.json')
    ]
    for files in train_test_files:
        for f in files:
            if not os.path.isfile(f'data/{f}'):
                raise ValueError(f'Missing a file"{f}"')

    for train, test_first, test_joined in tqdm.tqdm(train_test_files):
        experiment(train, test_first, test_joined)


if __name__ == '__main__':
    main()
