# data represented is this (only indexed, batched, and embedded)
# Word + confidences (split up e.g. "the first section" and ".68, 1, 1":  [['the', 0.6890783], ['first', 1], ['section', 1], ['is', 0.9697481], ['comprised', 1], ['of', 0.9999964], ['<unk>', 1]]
# page:  republic_(plato))
# qnum: 10329
# sentence #: usually 0-6

from dataset import QuizBowl
import torch

def eval(model, test_data):  #but test_data is different for all 3 people.  Mine and Pedro's can be merged (and he can not use confidences)
    # 1) load model: pickl?

    # 2) load needed data
    #test_data = (test_data)  # same for me and Pedro, although I'll use confidences.  Has qnum, sentence #,
                                        # answer, and decoded sentence at word level

    train_iter, val_iter, dev_iter = QuizBowl.iters( batch_size=512,
                                                            lower= True,
                                                            use_wiki=False,  #irrelevant
                                                            n_wiki_sentences=5, #irrelevant
                                                            replace_title_mentions='',
                                                            combined_ngrams=True,
                                                            unigrams=True,
                                                            bigrams=False, #irrelevant 
                                                            trigrams=False, #irrelevant 
                                                            combined_max_vocab_size=300000,
                                                            unigram_max_vocab_size= None, 
                                                            bigram_max_vocab_size=50000, #irrelevant 
                                                            trigram_max_vocab_size=50000 #irrelevant 
                                                            )
    print(len(dev_iter))


    #     # 3) loop through iterator's batches
    for batch in dev_iter:
        text, lengths = batch.text
        # create appropriate dict for input_ASR
        # run model, SEE BELOW FOR EACH PERSON.
        _, preds = torch.max(out, 1)
        accuracy = torch.mean(torch.eq(preds, page).float()).data[0]
        batch_loss = loss_function(out, page)

        print+save(avg(batch_accuracies))
        print+save(avg(batch_losses))

if __name__ == "__main__":
    eval()



#
#     # 4) plotting/visualization
#
#
# # Run Model is different for each person since inputs are different
# # Denis
# ##(where words are either top-1 predictions or all precitions + respective confidences)
# #out = model(input_ASR, lengths, qnums, confidences)  #lengths needed due to torchtext
# #...
#
# # Pedro
# out = model(input_ASR, lengths, qnums)
# ...
#
# # Joe
# #out = model(input_lattices, lengths, qnums(?))
# ...

