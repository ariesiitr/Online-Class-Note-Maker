import pdfplumber

import summarizer as sum
import caption as cap
import PyPDF2
import fitz
import io
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
file = '2.pdf'
pdfFileObj = open(file, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# open the file
pdf_file = fitz.open(file)

#textfile


x = (pdfReader.numPages)
with pdfplumber.open(file) as pdf:
    for i in range(0,x):
        first_page = pdf.pages[i]
        text,summ=sum.summerize(first_page.extract_text())
        page = pdf_file[i]
        image_list = page.getImageList()
        # printing number of images found in this page
        captions=[]
        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            im = Image.open(io.BytesIO(image_bytes))
            im = cap.encodeImage(im).reshape((1,2048))
            captions.append(cap.generateCaption(im))
            
        print("*************************")
        print("Page No. ", i+1)
        print("Text :")
        print(text)
        print("Summary :")
        print(summ)
        for j in captions:
         print("Caption :", j)
        print("*************************")
