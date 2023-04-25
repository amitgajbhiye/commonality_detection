import csv
import os

import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from torchtext.data import get_tokenizer
from gensim_modelling import GloveVectorsGensim

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

        self.tokenizer = get_tokenizer("basic_english")
        self.stop_words = stopwords.words("english")

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence_len = 0

        # sentence = sentence.translate(str.maketrans("", "", string.punctuation))

        tokenised_text = self.tokenizer(sentence)
        # tokenised_text = sentence.split()

        # tokenised_text = [
        #     word for word in tokenised_text if word not in self.stop_words
        # ]

        for word in tokenised_text:
            sentence_len += 1
            self.add_word(word)

        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len

        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def write_word_count_to_file(self, file_name):
        sorted_word2count = {
            k: v
            for k, v in sorted(
                self.word2count.items(), key=lambda item: item[1], reverse=True
            )
        }

        with open(file_name, "w") as out_file:
            w = csv.writer(out_file, delimiter="\t")
            w.writerows(sorted_word2count.items())

        print(f"Wiki Words count written to file : {file_name}")


def make_vocab(vocab, file_name, num_sent_to_process=None):
    sent_counter = 0

    with open(file_name, "r") as in_f:
        for sent in in_f:
            sent = sent.strip()
            # print(sent)
            vocab.add_sentence(sent)

            sent_counter += 1

            if num_sent_to_process and (sent_counter > num_sent_to_process):
                print(
                    f"Processed {num_sent_to_process} sentences; Stopping Building Vocab ..."
                )
                break


def get_glove_words_count(w2v_glove_file, wiki_vocab, out_glove_wiki_count_file):
    gv = GloveVectorsGensim(wv_format_glove_file=w2v_glove_file)
    glove_vocab = gv.glove_model.key_to_index.keys()

    glove_vocab_set = set(glove_vocab)
    wiki_vocab_set = set(wiki_vocab.word2count.keys())

    glove_wiki_words = glove_vocab_set.intersection(wiki_vocab_set)
    wiki_words_not_in_glove = wiki_vocab_set.difference(glove_vocab_set)

    with open("datasets/wiki_words_not_in_glove.txt", "w") as file:
        file.write("\n".join(list(wiki_words_not_in_glove)))

    print(flush=True)
    print(f"num_glove_words : {len(glove_vocab_set)}", flush=True)
    print(f"num_wiki_words: {len(wiki_vocab_set)}", flush=True)
    print(
        f"num_glove_wiki_words: {len(glove_wiki_words)}; Words common in Glove and Wiki",
        flush=True,
    )
    print(flush=True)
    print(
        f"num_wiki_words_not_in_glove : {len(wiki_words_not_in_glove)}; Words that are in Wiki but not in Glove",
        flush=True,
    )
    print(flush=True)

    glove_wiki_word_counts = {k: wiki_vocab.word2count[k] for k in glove_wiki_words}

    sorted_glove_wiki_word_counts = {
        k: v
        for k, v in sorted(
            glove_wiki_word_counts.items(), key=lambda item: item[1], reverse=True
        )
    }

    with open(out_glove_wiki_count_file, "w") as out_file:
        w = csv.writer(out_file, delimiter="\t")
        w.writerows(sorted_glove_wiki_word_counts.items())

    print(f"Glove Words Wiki count written to file : {out_glove_wiki_count_file}")


def main():
    print("Building English Wikipedia Vocabulary")

    # file_name = "/home/amitgajbhiye/Downloads/dummy_en_wikipedia.txt"
    # w2v_format_glove_file = "/home/amitgajbhiye/Downloads/embeddings_con_prop/glove.42B.300d.word2vec.format.txt"

    file_name = "/scratch/c.scmag3/en_wikipedia/en_wikipedia.txt"
    w2v_format_glove_file = "/scratch/c.scmag3/glove/glove.42B.300d.word2vec.format.txt"

    vocab = Vocabulary("en_wikipedia")

    make_vocab(vocab=vocab, file_name=file_name, num_sent_to_process=None)

    print("num_wiki_words: Number of Wiki Words", flush=True)
    print(len(vocab.word2count.keys()))

    vocab.write_word_count_to_file("datasets/word_counts_en_wikipedia_pytorch_tok.txt")

    out_glove_wiki_count_file = "datasets/glove_words_wiki_count_pytorch_tok.txt"
    get_glove_words_count(
        w2v_glove_file=w2v_format_glove_file,
        wiki_vocab=vocab,
        out_glove_wiki_count_file=out_glove_wiki_count_file,
    )


if __name__ == "__main__":
    main()
