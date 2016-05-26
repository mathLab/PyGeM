"""
Utilities for handling the Graphic Unit Interface.

.. todo::
	Switch to Ttk instead of Tk for a better look of the GUI
"""

import Tkinter
from tkFileDialog import askopenfilename
from PIL import ImageTk, Image
import pygem as pg
import sys
import os
import webbrowser

class Gui(object):
	"""
	The class for the Graphic Unit Interface.

	:cvar string filename_geometry: input geometry to be morphed.
	:cvar string filename_parameters: input parameters file for FFD.
	:cvar int check_var_1: dump or not the original FFD lattice.
	:cvar int check_var_2: dump or not the morphed FFD lattice.
	:cvar string outfilename: name of the output file geometry. 
		The extension of the file is set automatically equal to the on of input file 'filename_geometry'.
	:cvar string outfilename_lattice_orig: name of the dumped file for the original lattice.
		The extension of the file is set automatically equal to '.vtk'.
	:cvar string outfilename_lattice_mod: name of the dumped file for the morphed lattice.
		The extension of the file is set automatically equal to '.vtk'.	
	:cvar Tkinter.Tk root: main window object of the GUI.
	:cvar string print_geometry_path: geometry path to be printed close to the 'pick geometry' button.
	:cvar string print_parameter_path: parameters file path to be printed close to the 'pick parameters' button.
	:cvar Tkinter.Label label_geo: label related to 'print_geometry_path'.
	:cvar Tkinter.Label label_params: label related to 'print_parameters_path'.
	:cvar string url: url of the github page of PyGeM.
	
	"""
	
	def __init__(self):
	
		self.root = Tkinter.Tk()
		self.root.title('PyGeM')
		
		self.filename_geometry = Tkinter.StringVar()
		self.filename_parameters = Tkinter.StringVar()
		self.check_var_1 = Tkinter.IntVar()
		self.check_var_2 = Tkinter.IntVar()
		self.outfilename = Tkinter.StringVar()
		self.outfilename_lattice_orig = Tkinter.StringVar()
		self.outfilename_lattice_mod = Tkinter.StringVar()
		self.print_geometry_path = Tkinter.StringVar()
		self.print_parameter_path = Tkinter.StringVar()
		self.label_geo = None
		self.label_params = None
		self.url = 'https://github.com/mathLab/PyGeM'
		
		
		
	def _chose_geometry(self):
		"""
		The private method explores the file system and allows to select the wanted geometry.
		Up to now, you can select only IGES, OpenFOAM, STL, UNV or VTK geometry file.
		"""
		self.filename_geometry = askopenfilename(filetypes=[("IGES File",('*.iges', '*.igs')), \
		("OpenFOAM File",'*'),('STL File','*.stl'),('UNV File','*.unv'),('VTK File','*.vtk'),('All','*')])
		self.print_geometry_path.set(self.filename_geometry)
		self.label_geo.configure(fg='green')
		
	
	def _chose_parameters(self):
		"""
		The private method explores the file system and allows to select the wanted parameters file.
		It visualizes only .prm files.
		"""
		self.filename_parameters = askopenfilename(filetypes=[("Params File","*.prm")])
		self.print_parameter_path.set(self.filename_parameters)
		self.label_params.configure(fg='green')
		

	def _run_simulation(self):
		"""
		The private method runs the geometrical morphing.
		"""
		params = pg.params.FFDParameters()
		params.read_parameters(filename=self.filename_parameters)

		(__,file_extension_in) = os.path.splitext(self.filename_geometry)
			
		if file_extension_in == '.stl':
			geo_handler = pg.stlhandler.StlHandler()
		elif file_extension_in in ['.iges', '.igs']:
			geo_handler = pg.igeshandler.IgesHandler()
		elif file_extension_in == '.unv':
			geo_handler = pg.unvhandler.UnvHandler()
		elif file_extension_in == '':
			geo_handler = pg.openfhandler.OpenFoamHandler()
		elif file_extension_in == '.vtk':
			geo_handler = pg.vtkhandler.VtkHandler()
		else:
			raise NotImplementedError("Format not implemented yet")
			
		mesh_points = geo_handler.parse(self.filename_geometry)

		free_form = pg.freeform.FFD(params, mesh_points)
		free_form.perform()
		new_mesh_points = free_form.modified_mesh_points

		geo_handler.write(new_mesh_points, self.outfilename.get() + file_extension_in)

		if self.check_var_1.get() == 1:
			pg.utils.write_bounding_box(params, self.outfilename_lattice_orig.get() + '.vtk', False)
		if self.check_var_2.get() == 1:
			pg.utils.write_bounding_box(params, self.outfilename_lattice_mod.get() + '.vtk', True)
				
		
	def _goto_website(self):
		"""
		The private method opens the PyGeM main page on github. 
		It is used for info about PyGeM in the menu.
		"""
		webbrowser.open(self.url)
	
	
	def start(self):
		"""
		The method inizializes and visualizes the window.
		"""

		image = Image.open('readme/logo_PyGeM_small.png')
		image = image.resize((50, 50), Image.ANTIALIAS)
		img = ImageTk.PhotoImage(image)
		panel = Label(self.root, image = img)
		panel.pack(side = "bottom", padx = 5, pady = 5,anchor=SE)

		geo_frame = Frame(self.root)
		geo_frame.pack(anchor=W)

		# Buttons 1
		button_1 = Tkinter.Button(geo_frame, text ="Pick the geometry", command = self._chose_geometry)
		button_1.pack(side=LEFT, padx = 5, pady = 5)
		self.label_geo=Label(geo_frame, textvariable=self.print_geometry_path, fg='red')
		self.print_geometry_path.set("No geometry chosen!")
		self.label_geo.pack(side=LEFT, padx = 5, pady = 5)

		# Button 2
		params_frame = Frame(self.root)
		params_frame.pack(anchor=W)
		
		button_2 = Tkinter.Button(params_frame, text ="Pick the parameters", command = self._chose_parameters)
		button_2.pack(side=LEFT, padx = 5, pady = 5)
		self.label_params = Label( params_frame, textvariable=self.print_parameter_path, fg='red')
		self.print_parameter_path.set("No parameters file chosen!")
		self.label_params.pack(side=LEFT, padx = 5, pady = 5)

		# Entry
		entryframe = Frame(self.root)
		entryframe.pack(padx = 5, pady = 5, anchor=W)

		label_geo_out = Label(entryframe, text="Output geometry file")
		label_geo_out.pack( side = LEFT)
		entry_geo_out = Entry(entryframe, bd =5, textvariable=self.outfilename)
		entry_geo_out.pack(side = LEFT)

		# Checkboxes
		checkframe_orig = Frame(self.root)
		checkframe_orig.pack(anchor=W)
		
		check_lattice_orig = Checkbutton(checkframe_orig, text = "Dump Original FFD lattice", variable = self.check_var_1, \
				         onvalue = 1, offvalue = 0, height=3, \
				         width = 20)
				         
		check_lattice_orig.pack(side=LEFT)
		
		entry_lattice_orig = Entry(checkframe_orig, bd =5, textvariable=self.outfilename_lattice_orig)
		entry_lattice_orig.pack(side = LEFT)
		
		checkframe_mod = Frame(self.root)
		checkframe_mod.pack(anchor=W)         
				         
		check_lattice_mod = Checkbutton(checkframe_mod, text = "Dump Morphed FFD lattice", variable = self.check_var_2, \
				         onvalue = 1, offvalue = 0, height=3, \
				         width = 20)
		
		check_lattice_mod.pack(side=LEFT)
		
		entry_lattice_mod = Entry(checkframe_mod, bd =5, textvariable=self.outfilename_lattice_mod)
		entry_lattice_mod.pack(side = LEFT)
		
		# Run button
		button_run = Tkinter.Button(self.root, text ="Run PyGeM", command = self._run_simulation, bg='#065893', fg='#f19625', font='bold')
		button_run.pack()

		# Menu
		menubar = Menu(self.root)
		
		helpmenu = Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About...", command=self._goto_website)
		menubar.add_cascade(label="Help", menu=helpmenu)

		self.root.config(menu=menubar)

		self.root.mainloop()

