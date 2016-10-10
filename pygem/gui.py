"""
Utilities for handling the Graphic Unit Interface.

.. todo::
	Switch to Ttk instead of Tk for a better look of the GUI
"""
try:
	import tkinter
	from tkinter.filedialog import askopenfilename
except:
	import Tkinter as tkinter
	from tkFileDialog import askopenfilename

import os
import webbrowser
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Gui(object):
	"""
	The class for the Graphic Unit Interface.

	:cvar string filename_geometry: input geometry to be morphed.
	:cvar string filename_parameters: input parameters file for FFD.
	:cvar int check_var_dump_orig: dump or not the original FFD lattice.
	:cvar int check_var_dump_morphed: dump or not the morphed FFD lattice.
	:cvar string outfilename: name of the output file geometry. 
		The extension of the file is set automatically equal to the on of input file 'filename_geometry'.
	:cvar string outfilename_lattice_orig: name of the dumped file for the original lattice.
		The extension of the file is set automatically equal to '.vtk'.
	:cvar string outfilename_lattice_mod: name of the dumped file for the morphed lattice.
		The extension of the file is set automatically equal to '.vtk'. 
	:cvar tkinter.Tk root: main window object of the GUI.
	:cvar string print_geometry_path: geometry path to be printed close to the 'pick geometry' button.
	:cvar string print_parameter_path: parameters file path to be printed close to the 
		'pick parameters' button.
	:cvar tkinter.Label label_geo: label related to 'print_geometry_path'.
	:cvar tkinter.Label label_params: label related to 'print_parameters_path'.
	:cvar string url: url of the github page of PyGeM.
	:cvar tkinter.Canvas logo_panel: canvas for PyGeM logo.
	:cvar tkinter.PhotoImage img: PyGeM logo.
	:cvar tkinter.Frame orig_geo_frame: frame for plotting of the original geometry.
	:cvar tkinter.Frame mod_geo_frame: frame for plotting of the final geometry.
	"""

	def __init__(self):
		self.root = tkinter.Tk()
		self.root.resizable(width=False, height=False)
		self.root.minsize(width=1400, height=400)
		self.root.maxsize(width=1400, height=400)
		self.root.title('PyGeM')
		self.filename_geometry = tkinter.StringVar()
		self.filename_parameters = tkinter.StringVar()
		self.check_var_dump_orig = tkinter.IntVar()
		self.check_var_dump_morphed = tkinter.IntVar()
		self.outfilename = tkinter.StringVar()
		self.outfilename_lattice_orig = tkinter.StringVar()
		self.outfilename_lattice_mod = tkinter.StringVar()
		self.print_geometry_path = tkinter.StringVar()
		self.print_parameter_path = tkinter.StringVar()
		self.label_geo = None
		self.label_params = None
		self.url = 'https://github.com/mathLab/PyGeM'
		self.logo_panel = None
		self.img = None
		self.orig_geo_frame = None
		self.mod_geo_frame = None

	def _chose_geometry(self):
		"""
		The private method explores the file system and allows to select the wanted geometry.
		Up to now, you can select only IGES, OpenFOAM, STL, UNV or VTK geometry file.
		"""
		self.filename_geometry = askopenfilename(filetypes=[("IGES File", ('*.iges', '*.igs')), \
		 ("OpenFOAM File", '*'), ('STL File', '*.stl'), ('UNV File', '*.unv'), \
		 ('VTK File', '*.vtk'), ('All', '*')])
		self.print_geometry_path.set(self.filename_geometry)
		self.label_geo.configure(fg='green')

	def _chose_parameters(self):
		"""
		The private method explores the file system and allows to select the wanted parameters file.
		It visualizes only .prm files.
		"""
		self.filename_parameters = askopenfilename(
			filetypes=[("Params File", "*.prm")]
		)
		self.print_parameter_path.set(self.filename_parameters)
		self.label_params.configure(fg='green')

	def _run_simulation(self):
		"""
		The private method runs the geometrical morphing.
		"""
		import pygem as pg
		params = pg.params.FFDParameters()
		params.read_parameters(filename=self.filename_parameters)

		file_extension_in = os.path.splitext(self.filename_geometry)[-1]
		ext_handlers = {'.stl': pg.stlhandler.StlHandler(), \
		 '.iges': pg.igeshandler.IgesHandler(), \
		 '.igs': pg.igeshandler.IgesHandler(), \
		 '.unv': pg.unvhandler.UnvHandler(), \
		 '': pg.openfhandler.OpenFoamHandler(), \
		 '.vtk': pg.vtkhandler.VtkHandler()}

		if file_extension_in in ext_handlers:
			geo_handler = ext_handlers[file_extension_in]
		else:
			raise NotImplementedError("Format not implemented yet.")

		mesh_points = geo_handler.parse(self.filename_geometry)
		free_form = pg.freeform.FFD(params, mesh_points)
		free_form.perform()
		new_mesh_points = free_form.modified_mesh_points

		geo_handler.write(
			new_mesh_points, self.outfilename.get() + file_extension_in
		)

		if self.check_var_dump_orig.get() == 1:
			pg.utils.write_bounding_box(
				params, self.outfilename_lattice_orig.get() + '.vtk', False
			)
		if self.check_var_dump_morphed.get() == 1:
			pg.utils.write_bounding_box(
				params, self.outfilename_lattice_mod.get() + '.vtk', True
			)

		if file_extension_in in ['.vtk', '.stl', '.iges', '.igs']:
			figure_in = geo_handler.plot()
			figure_in.set_size_inches(4, 3)
			FigureCanvasTkAgg(figure_in, master=self.orig_geo_frame).get_tk_widget().grid(row=1, column=0, \
			 padx=5, pady=5)
			figure_out = geo_handler.plot(
				self.outfilename.get() + file_extension_in
			)
			figure_out.set_size_inches(4, 3)
			FigureCanvasTkAgg(figure_out, master=self.mod_geo_frame).get_tk_widget().grid(row=1, column=0, \
			 padx=5, pady=5)

	def _goto_website(self):
		"""
		The private method opens the PyGeM main page on github. 
		It is used for info about PyGeM in the menu.
		"""
		webbrowser.open(self.url)

	def _main(self):
		"""
		The private method inizializes and visualizes the window.
		"""
		self.logo_panel = tkinter.Canvas(self.root, height=60, width=60)
		self.logo_panel.pack(side="bottom", padx=5, pady=5, anchor=tkinter.SE)
		try:
			self.img = tkinter.PhotoImage(
				master=self.logo_panel, file='readme/logo_PyGeM_gui.gif'
			)
		except:
			self.img = tkinter.PhotoImage(
				master=self.logo_panel, file='../readme/logo_PyGeM_gui.gif'
			)
		self.logo_panel.create_image(35, 35, image=self.img)

		self.orig_geo_frame = tkinter.Frame(
			self.root, height=450, width=360, bg='#c1d0f0'
		)
		self.orig_geo_frame.pack(side="left", padx=5, pady=5)
		self.orig_geo_frame.pack_propagate(0)
		tkinter.Label(self.orig_geo_frame, text="INPUT GEOMETRY", bg='#c1d0f0', \
		 font=("Arial", 20)).grid(row=0, column=0, padx=3, pady=3)

		self.mod_geo_frame = tkinter.Frame(
			self.root, height=450, width=360, bg='#80ff80', padx=5, pady=5
		)
		self.mod_geo_frame.pack(side="right", padx=5, pady=5)
		self.mod_geo_frame.pack_propagate(0)
		tkinter.Label(self.mod_geo_frame, text="OUTPUT GEOMETRY", bg='#80ff80', \
		 font=("Arial", 20)).grid(row=0, column=0, padx=3, pady=3)

		code_frame = tkinter.Frame(
			self.root,
			height=490,
			width=360,
			relief=tkinter.GROOVE,
			borderwidth=1
		)
		code_frame.pack(padx=5, pady=5)
		code_frame.pack_propagate(0)

		# Buttons 1
		tkinter.Button(code_frame, text="Pick the geometry", \
		 command=self._chose_geometry).grid(row=0, column=0, padx=3, pady=3)
		self.label_geo = tkinter.Label(
			code_frame, textvariable=self.print_geometry_path, fg='red'
		)
		self.print_geometry_path.set("No geometry chosen!")
		self.label_geo.grid(row=0, column=1, padx=3, pady=3)

		# Button 2
		tkinter.Button(code_frame, text="Pick the parameters", \
		 command=self._chose_parameters).grid(row=1, column=0, padx=3, pady=3)
		self.label_params = tkinter.Label(
			code_frame, textvariable=self.print_parameter_path, fg='red'
		)
		self.print_parameter_path.set("No parameters file chosen!")
		self.label_params.grid(row=1, column=1, padx=3, pady=3)

		# Entry
		tkinter.Label(
			code_frame, text="Output geometry file"
		).grid(
			row=2, column=0, padx=3, pady=3
		)
		tkinter.Entry(code_frame, bd=5, textvariable=self.outfilename).grid(row=2, column=1, \
		 padx=3, pady=3)

		# Checkboxes
		tkinter.Checkbutton(code_frame, text="Dump Original FFD lattice", \
		 variable=self.check_var_dump_orig, onvalue=1, offvalue=0, height=3, width=20).grid(row=3, \
		 column=0)
		tkinter.Entry(
			code_frame, bd=5, textvariable=self.outfilename_lattice_orig
		).grid(
			row=3, column=1
		)

		tkinter.Checkbutton(code_frame, text="Dump Morphed FFD lattice", \
		 variable=self.check_var_dump_morphed, onvalue=1, offvalue=0, height=3, width=20).grid(row=4, \
		 column=0)
		tkinter.Entry(
			code_frame, bd=5, textvariable=self.outfilename_lattice_mod
		).grid(
			row=4, column=1
		)

		# Run button
		tkinter.Button(code_frame, text="Run PyGeM", command=self._run_simulation, bg='#065893', \
		 fg='#f19625', font='bold').grid(row=5, column=0, columnspan=2, padx=3, pady=3)

		# Menu
		menubar = tkinter.Menu(self.root)
		helpmenu = tkinter.Menu(menubar, tearoff=0)
		helpmenu.add_command(label="About...", command=self._goto_website)
		menubar.add_cascade(label="Help", menu=helpmenu)

		self.root.config(menu=menubar)

	def start(self):
		"""
		This method inizializes and starts the GUI. 
		"""
		self._main()
		self.root.mainloop()


if __name__ == "__main__":
	app = Gui()
	app.start()
