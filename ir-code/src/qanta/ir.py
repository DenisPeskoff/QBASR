from typing import List, Optional, Dict
import pprint
from collections import defaultdict
import json
import os
import subprocess

import tqdm
import click
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from elasticsearch_dsl.connections import connections
import elasticsearch
from jinja2 import Environment, PackageLoader
from nltk.tokenize import word_tokenize
from flask import jsonify, Flask, request

# from qanta.wikipedia import Wikipedia
from qanta.util import safe_path, load_config
from qanta import qlogging


ES_PARAMS = 'es_params.pickle'
connections.create_connection(hosts=['localhost'])


log = qlogging.get(__name__)


class Document:
    def __init__(self, page: str,
                 qb: Optional[str] = None,
                 asr: Optional[str] = None,
                 wiki: Optional[str] = None):
        self.page = page
        self.qb = qb
        self.asr = asr
        self.wiki = wiki


def merge_asr_sentences(sentences):
    text = []
    for sent in sentences:
        text.append(' '.join(w for w in sent if w != '<unk>'))
    return '. '.join(text)


class IrDataset:
    def __init__(self, fold: str):
        self._fold = fold

    def read(self, file_path, asr_path=None, jeopardy=False) -> List[Document]:
        if jeopardy:
            questions = []
            log.info(f'Reading instances from {asr_path}')
            with open(asr_path) as f:
                for q in json.load(f)['questions']:
                    text = merge_asr_sentences(q['sentences'])
                    page = q['page']
                    questions.append(self.text_to_instance(text, asr_text=text, page=page))
            return questions

        if asr_path is not None:
            log.info(f'Reading instances from {asr_path}')
            with open(asr_path) as f:
                asr_dataset = json.load(f)['questions']
                asr_questions = {q['qnum']: merge_asr_sentences(q['sentences']) for q in asr_dataset}
        else:
            asr_questions = {}

        log.info(f'Reading instances in fold="{self._fold}" from {file_path}')
        with open(file_path) as f:
            questions = []
            for q in json.load(f)['questions']:
                if q['page'] is not None and q['fold'] == self._fold:
                    qanta_id = q['qanta_id']
                    text = q['text']
                    page = q['page']
                    if qanta_id in asr_questions:
                        asr_text = asr_questions[qanta_id]
                    else:
                        asr_text = ''
                    questions.append(self.text_to_instance(text, asr_text=asr_text, page=page))
            return questions

    def text_to_instance(self, text: str, asr_text=None, page=None) -> Document:
        return Document(
            page,
            qb=text,
            asr=asr_text,
            wiki=None
        )


def create_es_config(output_path, host='localhost', port=9200):
    data_dir = safe_path('elasticsearch/data/')
    log_dir = safe_path('elasticsearch/log/')
    env = Environment(loader=PackageLoader('qanta', 'templates'))
    template = env.get_template('elasticsearch.yml.jinja2')
    config_content = template.render({
        'host': host,
        'port': port,
        'log_dir': log_dir,
        'data_dir': data_dir
    })
    with open(output_path, 'w') as f:
        f.write(config_content)


def start_elasticsearch():
    config_dir = 'config/'
    pid_file = 'elasticsearch/pid'
    subprocess.run(
        ['elasticsearch', '-d', '-p', pid_file, f'-Epath.conf={config_dir}']
    )


def stop_elasticsearch():
    pid_file = 'elasticsearch/pid'
    with open(pid_file) as f:
        pid = int(f.read())
    subprocess.run(['kill', str(pid)])


def create_doctype(index_name, similarity):
    if similarity == 'default':
        wiki_content_field = Text()
        qb_content_field = Text()
        asr_content_field = Text()
    else:
        wiki_content_field = Text(similarity=similarity)
        qb_content_field = Text(similarity=similarity)
        asr_content_field = Text(similarity=similarity)

    class Answer(DocType):
        page = Text(fields={'raw': Keyword()})
        wiki_content = wiki_content_field
        qb_content = qb_content_field
        asr_content = asr_content_field

        class Meta:
            index = index_name

    return Answer


class IrGuesser:
    def __init__(self, use_qb=True, use_asr=True):
        self._use_qb = use_qb
        self._use_asr = use_asr
        self._config = load_config()['ir']
        self._use_wiki = self._config['use_wiki']
        self._norm_score = self._config['norm_score']
        self._similarity = self._config['similarity']
        self._k1 = self._config['k1']
        self._b = self._config['b']
        self._index = IrIndex(
            similarity=self._similarity,
            bm25_b=self._b, bm25_k1=self._k1
        )

    def train(self, training_data: List[Document]):
        log.info(f'Config:\n{pprint.pformat(self._config)}')
        qb_merged = defaultdict(list)
        asr_merged = defaultdict(list)
        page_set = set()
        for doc in training_data:
            page = doc.page
            page_set.add(page)
            if doc.qb is not None:
                qb_merged[page].append(doc.qb)
            if doc.asr is not None:
                asr_merged[page].append(doc.asr)

        qb_final = {}
        asr_final = {}
        for page in page_set:
            qb_final[page] = ' '.join(qb_merged[page])
            asr_final[page] = ' '.join(asr_merged[page])

        self._index.build(
            qb_final, asr_final,
            use_wiki=self._use_wiki, use_qb=self._use_qb, use_asr=self._use_asr,
            rebuild_index=True
        )

    def guess(self, questions: List[str], max_guesses=1):
        def query(text):
            return self._index.search(
                text, max_guesses, normalize_score_by_length=self._norm_score
            )
        return [query(q) for q in tqdm.tqdm(questions)]

    def api(self):
        app = Flask(__name__)

        @app.route('/api/1.0/quizbowl/status', methods=['GET'])
        def status():
            return jsonify({
                'batch': True,
                'batch_size': 500,
                'ready': True
            })

        @app.route('/api/1.0/quizbowl/act', methods=['POST'])
        def act():
            out = self.guess([request.json['text']])[0]
            return jsonify({
                'guess': out['guess'],
                'buzz': False
            })

        @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
        def batch_act():
            questions = [q['text'] for q in request.json['questions']]
            outs = self.guess(questions)
            return jsonify([
                {'guess': o['guess'], 'buzz': False}
                for o in outs
            ])

        return app


class IrIndex:
    def __init__(self, name='qb', similarity='default', bm25_b=None, bm25_k1=None):
        self.name = name
        self.ix = Index(self.name)
        self.answer_doc = create_doctype(self.name, similarity)
        if bm25_b is None:
            bm25_b = .75
        if bm25_k1 is None:
            bm25_k1 = 1.2
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1

    def delete(self):
        try:
            self.ix.delete()
        except elasticsearch.exceptions.NotFoundError:
            log.info('Could not delete non-existent index.')

    def exists(self):
        return self.ix.exists()

    def init(self):
        self.ix.create()
        self.ix.close()
        self.ix.put_settings(
            body={'similarity': {'qb_bm25': {'type': 'BM25', 'b': self.bm25_b, 'k1': self.bm25_k1}}}
        )
        self.ix.open()
        self.answer_doc.init(index=self.name)

    def build(self, qb_docs: Dict[str, str], asr_docs: Dict[str, str],
              use_wiki=False, use_qb=True, use_asr=True,
              rebuild_index=False):
        if rebuild_index or bool(int(os.getenv('QB_REBUILD_INDEX', 0))):  # pylint: disable=invalid-envvar-default
            log.info(f'Deleting index: {self.name}')
            self.delete()

        if self.exists():
            log.info(f'Index {self.name} exists')
        else:
            log.info(f'Index {self.name} does not exist')
            self.init()
            # wiki_lookup = Wikipedia()
            log.info('Indexing...')
            for page in tqdm.tqdm(qb_docs):
                wiki_content = ''
                # if use_wiki and page in wiki_lookup:
                #     wiki_content = wiki_lookup[page].text
                # else:
                #     wiki_content = ''

                if use_qb:
                    qb_content = qb_docs[page]
                else:
                    qb_content = ''

                if use_asr:
                    asr_content = asr_docs[page]
                else:
                    asr_content = ''

                answer = self.answer_doc(
                    page=page,
                    wiki_content=wiki_content,
                    qb_content=qb_content,
                    asr_content=asr_content
                )
                answer.save(index=self.name)

    def search(self, text: str, max_n_guesses: int, normalize_score_by_length=False):
        if not self.exists():
            raise ValueError('The index does not exist, you must create it before searching')

        wiki_field = 'wiki_content'
        qb_field = 'qb_content'
        asr_field = 'asr_content'

        s = Search(index=self.name)[0:max_n_guesses].query(  # pylint: disable=no-member
            'multi_match', query=text, fields=[wiki_field, qb_field, asr_field]
        )
        results = s.execute()
        guess_set = set()
        guesses = []
        if normalize_score_by_length:
            query_length = max(1, len(text.split()))
        else:
            query_length = 1

        for r in results:
            if r.page in guess_set:
                continue
            else:
                guesses.append({'guess': r.page, 'score': r.meta.score, 'length': query_length})
        if len(guesses) == 0:
            return {'guess': '~~~NOGUESS~~~', 'score': 0, 'length': 1}
        else:
            return guesses[0]


@click.group(name='ir')
def cli():
    pass


@cli.command()
def start():
    log.info('Creating config')
    create_es_config('config/elasticsearch.yml')
    log.info('Starting elastic search')
    start_elasticsearch()


@cli.command()
def stop():
    log.info('Stopping elastic search')
    stop_elasticsearch()


@cli.command(name='train')
@click.option('--train-data', default='../data/qanta.train.2018.04.18.json')
@click.option('--asr-data', default='../data/asr_qanta.train.nounk.2018.04.18.json')
@click.option('--clean/--no-clean', default=True)
@click.option('--asr/--no-asr', default=True)
@click.option('--jeopardy', default=False, is_flag=True)
def train(train_data, asr_data, clean, asr, jeopardy):
    log.info(f'ASR Train data: {asr_data}')
    log.info(f'asr={asr} clean={clean}')
    dataset = IrDataset('guesstrain').read(
        train_data, asr_path=asr_data, jeopardy=jeopardy
    )
    log.info('Building index')
    guesser = IrGuesser(use_qb=clean, use_asr=asr)
    guesser.train(dataset)


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
def web(host, port):
    guesser = IrGuesser()
    app = guesser.api()
    app.run(host=host, port=port, debug=False)


@cli.command()
@click.option('--titles-data', default='data/wikipedia-titles.2018.04.18.json')
@click.option('--eval-data', default='data/qanta.dev.nounk.2018.04.18.json')
@click.option('--skip-title-check', default=False, is_flag=True)
def asr_eval(titles_data, eval_data, skip_title_check):
    log.info(f'Eval data: {eval_data}')
    guesser = IrGuesser()
    answer_map = {
        'Thirty-six_Views_of_Mount_Fiji': 'Thirty-six_Views_of_Mount_Fuji'
    }
    with open(titles_data) as f:
        titles = set(json.load(f))
        lower_lookup = {t.lower(): t for t in titles}

    with open(eval_data) as f:
        ds_questions = json.load(f)['questions']
        first_questions = []
        full_questions = []
        pages = []
        all_full_guesses = []

        for q in ds_questions:
            if 'text' in q and 'first_sentence' in q:
                first_questions.append(q['first_sentence'])
                full_questions.append(q['text'])
            else:
                first_questions.append(merge_asr_sentences(q['sentences'][0:1]))
                full_questions.append(merge_asr_sentences(q['sentences']))

            cand_page = q['page']
            if not skip_title_check:
                cand_page = q['page'].replace(' ', '_')
                if cand_page not in titles:
                    if cand_page in answer_map:
                        cand_page = answer_map[cand_page]
                    elif cand_page.lower() not in lower_lookup:
                        raise ValueError(f'{cand_page} not in titles')
                    else:
                        cand_page = lower_lookup[cand_page.lower()]
            pages.append(cand_page)

        first_correct = 0
        first_guesses = guesser.guess(first_questions)
        for page, guess in zip(pages, first_guesses):
            if page == guess['guess']:
                first_correct += 1

        full_correct = 0
        full_guesses = guesser.guess(full_questions)
        for page, guess in zip(pages, full_guesses):
            all_full_guesses.append(guess)
            if page == guess['guess']:
                full_correct += 1

        total = len(pages)
        log.info(f'Eval Data: {eval_data}')
        log.info(f'First Accuracy: {first_correct / total}')
        log.info(f'Full  Accuracy: {full_correct / total}')

    with open(f'data/preds.json', 'w') as f:
        json.dump(all_full_guesses, f)
