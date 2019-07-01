#This is an evaluation script that loads a model, downloads data,

# data represented is this (only indexed, batched, and embedded)
# Word + confidences (split up e.g. "the first section" and ".68, 1, 1":  [['the', 0.6890783], ['first', 1], ['section', 1], ['is', 0.9697481], ['comprised', 1], ['of', 0.9999964], ['<unk>', 1]]
# page:  republic_(plato))
# qnum: 10329
# sentence #: usually 0-6

import dataset
import dataset_confidence
import torch
import importlib
importlib.reload(dataset)


def pull_data(ORIGINAL=True):
    if ORIGINAL:
        # original clean data
        train_iter, val_iter, dev_iter = dataset.QuizBowl.iters(batch_size=512,
                                                                lower=True,
                                                                use_wiki=False,  # irrelevant
                                                                n_wiki_sentences=5,  # irrelevant
                                                                replace_title_mentions='',
                                                                combined_ngrams=True,
                                                                unigrams=True,
                                                                bigrams=False,  # irrelevant
                                                                trigrams=False,  # irrelevant
                                                                combined_max_vocab_size=300000,
                                                                unigram_max_vocab_size=None,
                                                                bigram_max_vocab_size=50000,  # irrelevant
                                                                trigram_max_vocab_size=50000  # irrelevant
                                                                )
    # This handles confidences
    else:
        # somewhat different format.  Pulls different data and adds confidences
        train_iter, val_iter, dev_iter = dataset_confidence.QuizBowl.iters(batch_size=512,
                                                                           lower=True,
                                                                           use_wiki=False,  # irrelevant
                                                                           n_wiki_sentences=5,  # irrelevant
                                                                           replace_title_mentions='',
                                                                           combined_ngrams=True,
                                                                           unigrams=True,
                                                                           bigrams=False,  # irrelevant
                                                                           trigrams=False,  # irrelevant
                                                                           combined_max_vocab_size=300000,
                                                                           unigram_max_vocab_size=None,
                                                                           bigram_max_vocab_size=50000,
                                                                           # irrelevant
                                                                           trigram_max_vocab_size=50000
                                                                           # irrelevant
                                                                           )

    return dev_iter


def load_model():
    pass
    # pickle?


def eval_model(model):

    test_iter = pull_data(True)

    batch_accuracies = []
    batch_losses = []

    for batch in test_iter:
        input_dict = {}
        lengths_dict = {}
        if hasattr(batch, 'text'):
            text, lengths = batch.text
            input_dict['text'] = text
            lengths_dict['text'] = lengths
        page = batch.page
        qnums = batch.qanta_id  # or batch.qnum in asr currently
        out = model(input_dict, lengths_dict, qnums)
        _, preds = torch.max(out, 1)

        accuracy = torch.mean(torch.eq(preds, page).float()).data[0]
        batch_loss = loss_function(out, page)

        batch_accuracies.append(accuracy)
        batch_losses.append(batch_loss.data[0])

    return np.mean(batch_accuracies), np.mean(batch_losses)


if __name__ == "__main__":
    # model = load_model()
    output = eval_model(model)
    print(output)
