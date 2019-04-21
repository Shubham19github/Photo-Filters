import threading
import imutils
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import os

size = 15
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# kernel_sharpen = np.array([[1,1,1], [1,-7,1], [1,1,1]])

kernel_emboss_1 = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])

kernel_emboss_2 = np.array([[-1,-1,0],
                            [-1,0,1],
                            [0,1,1]])

kernel_emboss_3 = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])

class FilterApp(object):
	
	def __init__(self, vs):

		self.thread = None
		self.stopEvent = None
		self.vs = vs
		self.frame = None
		self.image = None
		self.panel = None
		self.currentfilter = None

		self.root = tk.Tk()
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onWindowClose)	#callback to window close function
		self.root.wm_title("Photo Filter")		#Title of window
		# self.root.geometry("600x600+200+100")	#Dimension of thee App
		self.root.config(bg="#fff")

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		self.addButtons()


	def changeFilter(self, filterValue):
		self.currentfilter = filterValue


	def addButtons(self):

		self.buttonframe = tk.Frame(self.root)
		self.buttonframe.pack(side="right", padx=10, pady=20)

		original = tk.Button(self.buttonframe, text="Original", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('original'))
		original.pack()

		gray = tk.Button(self.buttonframe, text="Gray", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('gray'))
		gray.pack()

		blur = tk.Button(self.buttonframe, text="Blur", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('blur'))
		blur.pack()

		sharp = tk.Button(self.buttonframe, text="Sharp", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('sharp'))
		sharp.pack()

		emboss = tk.Button(self.buttonframe, text="Emboss", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('emboss'))
		emboss.pack()

		vignette = tk.Button(self.buttonframe, text="Vignette", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('vignette'))
		vignette.pack()

		sketch = tk.Button(self.buttonframe, text="Sketch", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command= lambda: self.changeFilter('sketch'))
		sketch.pack()

		save = tk.Button(self.buttonframe, text="Save", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command=self.saveFilteredImage)
		save.pack()

		quit = tk.Button(self.buttonframe, text="Exit", bg="#000", fg="#fff", width=25, height=2, font='Helvetica 10 bold', activebackground="#fff", activeforeground="#000", padx=5, pady=5, command=self.onWindowClose)
		quit.pack()

	
	def videoLoop(self):
		try:
			while not self.stopEvent.is_set():
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=600)

				self.frame = self.affineTransform()

				if(self.currentfilter == 'gray'):
					self.frame = self.grayTransform()
				elif(self.currentfilter == 'blur'):
					self.frame = self.blurTransform()
				elif(self.currentfilter == 'sharp'):
					self.frame = self.sharpTransform()
				elif(self.currentfilter == 'emboss'):
					self.frame = self.embossTransform()
				elif(self.currentfilter == 'vignette'):
					self.frame = self.vignetteTransform()
				elif(self.currentfilter == 'sketch'):
					self.frame = self.sketchTransform()
				elif(self.currentfilter == 'original'):
					self.frame = self.frame

				self.image = self.frame
	
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)
		
				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tk.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image
 
		except RuntimeError:
			print("Exiting App")


	def affineTransform(self):

		rows, cols = self.frame.shape[:2]

		src_points = np.float32([ [0,0], [cols-1,0], [0, rows-1] ])
		dstn_points = np.float32([ [cols-1,0], [0,0], [cols-1,rows-1] ])

		affineMatrix = cv2.getAffineTransform(src_points, dstn_points)

		affineFrame = cv2.warpAffine(self.frame, affineMatrix, (cols,rows))

		return affineFrame

	def blurTransform(self):
		global kernel_motion_blur

		blurFrame = cv2.filter2D(self.frame, -1, kernel_motion_blur)

		return blurFrame

	def grayTransform(self):

		grayFrame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		return grayFrame

	def sharpTransform(self):

		global kernel_sharpen

		sharpFrame = cv2.filter2D(self.frame, -1, kernel_sharpen)

		return sharpFrame

	def embossTransform(self):

		global kernel_emboss_1, kernel_emboss_2, kernel_emboss_3

		grayImage = self.grayTransform()

		output_1 = cv2.filter2D(grayImage, -1, kernel_emboss_1)
		output_2 = cv2.filter2D(grayImage, -1, kernel_emboss_2)
		output_3 = cv2.filter2D(grayImage, -1, kernel_emboss_3)

		embossFrame = cv2.add(output_1, output_2, output_3)

		return embossFrame + 128

	def vignetteTransform(self):

		rows, cols = self.frame.shape[:2]

		# generating vignette mask using Gaussian kernels
		kernel_x = cv2.getGaussianKernel(cols,200)
		kernel_y = cv2.getGaussianKernel(rows,200)
		kernel = kernel_y * kernel_x.T
		mask = 255 * kernel / np.linalg.norm(kernel)
		vignetteFrame = np.copy(self.frame)

		# applying the mask to each channel in the input image

		for i in range(3):
			vignetteFrame[:,:,i] = vignetteFrame[:,:,i] * mask

		return vignetteFrame

	def sketchTransform(self):

		imageGray = self.grayTransform()

		imageGray = cv2.medianBlur(imageGray, 7)

		edges = cv2.Laplacian(imageGray, cv2.CV_8U, ksize=5)

		ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

		return mask


	def saveFilteredImage(self):

		if not os.path.exists('savedImages/'):
			os.makedirs('savedImages/')

		filename = "savedImages/{}.jpg".format(self.currentfilter)
		cv2.imwrite(filename, self.image)

	# on app close
	def onWindowClose(self):

		self.stopEvent.set()
		self.vs.stop()
		self.root.destroy()
