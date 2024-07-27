import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer#
from nltk.tokenize import word_tokenize#
nltk.download("punkt")

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    s =PorterStemmer()
    #print(ps.stem("drinked"))
    tk=ToktokTokenizer()
    tokens=tk.tokenize(preprocessed_text)
    tokens=[token.strip() for token in tokens]
    
    filter_words=[s.stem(w) for w in tokens]
    preprocessed_text=' '.join(filter_words)
    
    # End your code

    return preprocessed_text

"""
if __name__ == '__main__':
    text='programming programmer'
    a=remove_stopwords(text)
    print(a)
    b=preprocessing_function(text)
    print(b)
"""

