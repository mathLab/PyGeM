
from unittest import TestCase
import unittest
import pygem.gui as gui


class TestGui(TestCase):

	def test_gui_init_string_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.root.title() == 'PyGeM'
		
		
	def test_gui_init_string_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.filename_geometry.get() == ''
		
		
	def test_gui_init_string_3(self):
		gui_handler = gui.Gui()
		assert gui_handler.filename_parameters.get() == ''
		
		
	def test_gui_init_string_4(self):
		gui_handler = gui.Gui()
		assert gui_handler.outfilename.get() == ''
		
		
	def test_gui_init_string_5(self):
		gui_handler = gui.Gui()
		assert gui_handler.outfilename_lattice_orig.get() == ''
		
		
	def test_gui_init_string_6(self):
		gui_handler = gui.Gui()
		assert gui_handler.outfilename_lattice_mod.get() == ''
		
		
	def test_gui_init_string_7(self):
		gui_handler = gui.Gui()
		assert gui_handler.print_geometry_path.get() == ''
		
		
	def test_gui_init_string_8(self):
		gui_handler = gui.Gui()
		assert gui_handler.print_parameter_path.get() == ''
		
		
	def test_gui_init_string_9(self):
		gui_handler = gui.Gui()
		assert gui_handler.url == 'https://github.com/mathLab/PyGeM'
		
		
	def test_gui_init_int_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.check_var_1.get() == 0
		
		
	def test_gui_init_int_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.check_var_2.get() == 0
		
		
	def test_gui_init_none_1(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_geo == None
		
		
	def test_gui_init_none_2(self):
		gui_handler = gui.Gui()
		assert gui_handler.label_params == None
		
	
	def test_gui_init_none_3(self):
		gui_handler = gui.Gui()
		assert gui_handler.logo_panel == None
		
		
	def test_gui_init_none_4(self):
		gui_handler = gui.Gui()
		assert gui_handler.img == None
		
		
	def test_gui_init_none_5(self):
		gui_handler = gui.Gui()
		assert gui_handler.orig_geo_frame == None
		
		
	def test_gui_init_none_6(self):
		gui_handler = gui.Gui()
		assert gui_handler.mod_geo_frame == None
		

	def test_gui_init_all(self):
		gui.Gui()
		
		
	def test_gui_main(self):
		interface = gui.Gui()
		interface.main()
		
	
