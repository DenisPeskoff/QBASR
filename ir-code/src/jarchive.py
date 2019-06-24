import json


def main():
    with open('data/jeopardy.humanclean.json') as f:
        data = json.load(f)
        rounds = data['1'] + data['2'] + data['3']
        all_questions = []
        for r in rounds:
            questions = [
                {'text': q['prompt'], 'page': q['solution']}
                for q in r['questions']
                if q['Jtype'] != 'placeholder'
            ]
            for q in questions:
                q['sentences'] = [q['text'].split()]
            all_questions.extend(questions)
    with open('data/jeopardy.humanclean-formmatted.json', 'w') as f:
        json.dump({'questions': all_questions}, f)


if __name__ == '__main__':
    main()
