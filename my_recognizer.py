import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for index in range(test_set.num_items):
    
        word_scores = {}        
        seq, length = test_set.get_item_Xlengths(index)
        best_score, best_word = float("-inf"), None        

        for word, model in models.items():
            try:
                word_scores[word] = model.score(seq, length)
            except Exception as e:
                word_scores[word] = float("-inf")
            
            if word_scores[word] > best_score:
                best_score, best_word = word_scores[word], word
                
        probabilities.append(word_scores)
        guesses.append(best_word)
        
    return probabilities, guesses