import cherrypy, os, urllib
import cPickle as pickle
from numpy import *
import os
import zmq

"""
This is the image search demo in Section 7.6.
"""

class SearchDemo:
    def __init__(self):
	print "starting web server ... ..."
        # init the sockets
        print("Connecting to hello world server...")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

        self.imlist = os.listdir("thumbnails")
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)

        # set max number of results to show
        self.maxres = 100
        
        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search demo(on Ukbench dataset)</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """
        
    def index(self,query=None):
        
        html = self.header
        html += """
            <br />
            Created by Yang Zhou. Click an image to search. <a href='?query='> Random selection </a> of 10200 images in Ukbench dataset.
            <br /><br />
            """
        if query:
            # query the database and get top images
            self.socket.send(bytes(query))
            #  Get the reply.
            message = self.socket.recv()
            res = [int(x) for x in message.split()]
            for idx in res:
		imname = "ukbench%05d.th.jpg"%(idx)
		html += "<a href='?query="+imname+"'>"
		html += "<img src='"+imname+"' width='100' />"
		html += "</a>"
        else:
            # show random selection if no query
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='100' />"
                html += "</a>"
                
        html += self.footer
        return html
    
    index.exposed = True

cherrypy.quickstart(SearchDemo(), "", config=os.path.join(os.path.dirname(__file__), "service.conf"))
