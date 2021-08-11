from app import app
import os
import os.path



####################***SPACY LIB***####################
import spacy
from spacy.lang.ru.examples import sentences
from spacy.symbols import ORTH, LEMMA
from spacy.lang.ru import Russian
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
from spacy import displacy
import ru_core_news_lg