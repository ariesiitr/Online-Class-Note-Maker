import sumy
import pytesseract as pss
pss.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract'
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

def summerize(text):
    l = text.split('.')
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, len(l)//2)
    #print('**********************************')
    print("Text :")
    print(text)
    print("")
    print("Summary :")
    for sentence in lsa_summary:
        print(sentence)
    
def ocr(img):
    return pss.image_to_string(img)




