# this is for doing some math operations
import math
# this is for handling the PDF operations
import fitz
# importing PhotoImage from tkinter
from tkinter import PhotoImage



class PDFMiner:
    def __init__(self, filepath):
        # creating the file path
        self.filepath = filepath
        # opening the pdf document
        self.pdf = fitz.open(self.filepath)
        # loading the first page of the pdf document
        self.first_page = self.pdf.load_page(0)
        # getting the height and width of the first page
        self.width, self.height = self.first_page.rect.width, self.first_page.rect.height
        # initializing the zoom values of the page
        zoomdict = {800:0.8, 700:0.6, 600:1.0, 500:1.0}
        # getting the width value
        width = int(math.floor(self.width / 100.0) * 100)
        # zooming the page
        self.zoom = zoomdict[width]
        
    # this will get the metadata from the document like 
    # author, name of document, number of pages  
    def get_metadata(self):
        # getting metadata from the open PDF document
        metadata = self.pdf.metadata
        # getting number of pages from the open PDF document
        numPages = self.pdf.page_count
        # returning the metadata and the numPages
        return metadata, numPages
    
    # the function for getting the page
    def get_page(self, page_num):
        # loading the page
        page = self.pdf.load_page(page_num)
        # checking if zoom is True
        if self.zoom:
            # creating a Matrix whose zoom factor is self.zoom
            mat = fitz.Matrix(self.zoom, self.zoom)
            # gets the image of the page
            pix = page.get_pixmap(matrix=mat)
        # returns the image of the page  
        else:
            pix = page.get_pixmap()
        # a variable that holds a transparent image
        px1 = fitz.Pixmap(pix, 0) if pix.alpha else pix
        # converting the image to bytes
        imgdata = px1.tobytes("ppm")
        # returning the image data
        return PhotoImage(data=imgdata)
    
    
    # function to get text from the current page
    def get_text(self, page_num):
        # loading the page
        page = self.pdf.load_page(page_num)
        # getting text from the loaded page
        text = page.getText('text')
        # returning text
        return text