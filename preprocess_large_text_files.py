'''
Multi purpose pre processing script to handle large text files dump like wikipedia's. Usecase include:
1. General preprocessing.
2. combining logical bigram phrases as one token separated by '_'
3. combining logical trigram phrases as one token separated by '_'
4. Train on huge dump, save bigram ad trigram models and use them to transform small other text files.

Inspired by https://github.com/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb
'''
import os
import spacy
import codecs
import argparse
from tqdm import tqdm
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

# nlp = spacy.load("en_core_web_sm")

nlp = spacy.load('en')

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    
    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    
    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    
    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=4):
        for sent in tqdm(parsed_review.sents):
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])
    
def process_unigram(txt_filepath, unigram_txt_filepath):

    with codecs.open(unigram_txt_filepath, 'w', encoding='utf_8') as f:
        i = 0
        for sentence in lemmatized_sentence_corpus(txt_filepath):

            i = i + 1

            if (i % 10000 == 0):
                print('Unigram Processed ' + str(i) + ' articles')
            f.write(sentence + '\n')        

def train_bigram(unigram_txt_filepath, bigram_model_filepath, savebigram, bigram_txt_filepath):
    print('reading unigram text file.....')

    unigram_txt = LineSentence(unigram_txt_filepath)
    print('training bigram model.....')

    bigram_model = Phrases(unigram_txt)
    print('saving bigram model.....')

    bigram_model.save(bigram_model_filepath)
    
    # load the finished model from disk
    # bigram_model = Phrases.load(bigram_model_filepath)

    if savebigram:
        print('saving bigram processed text file....')

        with codecs.open(bigram_txt_filepath, 'w', encoding='utf_8') as f:
            i = 0
            for unigram_sentence in tqdm(unigram_txt):
                
                bigram_sentence = u' '.join(bigram_model[unigram_sentence])
                
                f.write(bigram_sentence + '\n')

                i = i + 1

                if (i % 10000 == 0):
                    print('Bigram Processed ' + str(i) + ' articles')

def train_trigram(bigram_txt_filepath, trigram_model_filepath, savetrigram, trigram_txt_filepath):

    print('reading bigram text file.....')

    bigram_txt = LineSentence(bigram_txt_filepath)

    print('training trigram model.....')

    trigram_model = Phrases(bigram_txt)

    print('saving trigram model.....')

    trigram_model.save(trigram_model_filepath)
        
    # load the finished model from disk
    # trigram_model = Phrases.load(trigram_model_filepath)


    if savetrigram:

        print('saving trigram processed text file....')
        with codecs.open(trigram_txt_filepath, 'w', encoding='utf_8') as f:
            i = 0
            for bigram_sentence in tqdm(bigram_txt):
                
                trigram_sentence = u' '.join(trigram_model[bigram_sentence])
                
                f.write(trigram_sentence + '\n')

                i = i + 1

                if (i % 10000 == 0):
                    print('Trigram Processed ' + str(i) + ' articles')

def train_corpus(txt_filepath, unigram_txt_filepath, includebigram, bigram_model_filepath, savebigram, bigram_txt_filepath, includetrigram, trigram_model_filepath, savetrigram, trigram_txt_filepath):

    if includebigram:

        process_unigram(txt_filepath, unigram_txt_filepath)
        train_bigram(unigram_txt_filepath, bigram_model_filepath, savebigram, bigram_txt_filepath)

        if includetrigram:
            train_trigram(bigram_txt_filepath, trigram_model_filepath, savetrigram, trigram_txt_filepath)


def transform_corpus(txt_filepath, output_txt_filepath, includebigram, bigram_model_filepath, includetrigram, trigram_model_filepath):
    
    if includebigram:

        if includetrigram:
            print('Transforming to trigrams...')
        else:
            print('Transforming to bigram....')

        with codecs.open(output_txt_filepath, 'w', encoding='utf_8') as f:
            i = 0
            for parsed_review in nlp.pipe(line_review(txt_filepath),
                                          batch_size=10000, n_threads=4):
                i = i + 1

                if (i % 10000 == 0):
                    print('Processed ' + str(i) + ' articles')

                # lemmatize the text, removing punctuation and whitespace
                unigram_review = [token.lemma_ for token in parsed_review
                                  if not punct_space(token)]
                
                # apply the first-order and second-order phrase models

                # load the bigram model from disk
                bigram_model = Phrases.load(bigram_model_filepath)
                bigram_review = bigram_model[unigram_review]
                 
                if includetrigram:
                    # load the trigram model from disk 
                    trigram_model = Phrases.load(trigram_model_filepath)
         
                    trigram_review = trigram_model[bigram_review]
                    
                    # remove any remaining stopwords
                    trigram_review = [term for term in trigram_review
                                      if term not in nlp.Defaults.stop_words]
                    
                    # write the transformed review as a line in the new file
                    trigram_review = u' '.join(trigram_review)
                    f.write(trigram_review + '\n')

                else:
                    # remove any remaining stopwords
                    bigram_review = [term for term in bigram_review
                                      if term not in nlp.Defaults.stop_words]

                    # write the transformed review as a line in the new file
                    bigram_review = u' '.join(bigram_review)
                    f.write(bigram_review + '\n')

    else:
        print('Transforming to unigram....')
        process_unigram(txt_filepath, output_txt_filepath)

if __name__ == '__main__':


     # Set up command line parameters.
    
    parser = argparse.ArgumentParser(description='Multipurpose pre processing file. Train bigram and trigram Phrase model or use existing models to transform input text file to output text file having bigrams and trigrams combined.')
    
    parser.add_argument('--inputfile', '-i', 
                        action='store',
                        default=None,
                        help=('Path to input text file.'))

    parser.add_argument('--outputfile', '-o', 
                        action='store',
                        default=None,
                        help=('Path to store processed text file'))

    parser.add_argument('--includebigram', '-b',
                        action='store_true',
                        default = False,
                        help=('Use to combine logical bigram phrases in the text.'))

    parser.add_argument('--includetrigram', '-t', action='store_true',
                        default = False,
                        help=('Use to combine logical trigram phrases in the text.'))

    parser.add_argument('--savebigram', '-sb', action='store_true',
                        default = True,
                        help=('Use to save bigram processed text file'))

    parser.add_argument('--savetrigram', '-st', action='store_true',
                        default = False,
                        help=('Use to save trigram processed text file'))

    parser.add_argument('--train', '-tr', action='store_true',
                        default = False,
                        help=('Train bigram and/or trigram Phrase model for input text file. Usage: python preprocess_large_text_files.py -tr -i <processed_text_file> -b -t -sb -st'))

    parser.add_argument('--transform', '-tf', action='store_true',
                        default = False,
                        help=('Transforms a text file using presaved bigram and/or trigram model. Usage: python preprocess_large_text_files.py -tf -i <processed_text_file> -o <output_file_path> -b -t'))

    parser.add_argument('--preprocess', '-pp', action='store_true',
                        default = False,
                        help=('Preprocesses a input text file, removes punctuation and lemmatises. Usage: python preprocess_large_text_files.py -pp -i <processed_text_file> -o <output_file_path>'))

    args = parser.parse_args()

    print('\n')
    print(args)

    txt_filepath = args.inputfile
    output_txt_filepath = args.outputfile

    unigram_txt_filepath = ('unigram_processed.txt')
    bigram_txt_filepath = ('bigram_processed.txt')
    trigram_txt_filepath = ('trigram_processed.txt')

    bigram_model_filepath = ('bigram_model')
    trigram_model_filepath = ('trigram_model')

    if args.train:
        
        print('Training......')
        train_corpus(txt_filepath, unigram_txt_filepath, args.includebigram, bigram_model_filepath, args.savetrigram, bigram_txt_filepath, args.includetrigram, trigram_model_filepath, args.savetrigram, trigram_txt_filepath)
        print('Training done... bigram and/or trigram model ready to use...')
        
    elif args.transform:

        print('Transforming.........')
        transform_corpus(txt_filepath, output_txt_filepath, args.includebigram, bigram_model_filepath, args.includetrigram, trigram_model_filepath)
        print('Transform done.. check processed output file')

    elif args.preprocess:

        print('Basic preprocessing of input text file')
        process_unigram(txt_filepath, output_txt_filepath)
        print('Preprocessing done... check output file')
