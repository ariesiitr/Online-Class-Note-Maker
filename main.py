import pdfplumber

import functions as fun
import caption as cap
import PyPDF2
import fitz
import io
from PIL import Image

pdfFileObj = open('2.pdf', 'rb')


pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
file = "2.pdf"
# open the file
pdf_file = fitz.open(file)

x = (pdfReader.numPages)
with pdfplumber.open(r'2.pdf') as pdf:
    for i in range(0,x):
        first_page = pdf.pages[i]
        print('**********************************')
        if None:
            print("Page No.",i+1)
            print("")
        else:
            print("Page No.",i+1)
            print("")
        fun.summerize(first_page.extract_text())
        page = pdf_file[i]
        image_list = page.getImageList()
        # printing number of images found in this page
        
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
            
            print("Caption ", image_index,":",cap.generateCaption(im))
            
            #print("---------------------")
            #print('**********************************')
            #im.show()
            # save it to local disk
            #im.save(open(f"/home/shreyansh/repo/Image-Caption/Images_extracted/image{i + 1}_{image_index}.{image_ext}", "wb"))

