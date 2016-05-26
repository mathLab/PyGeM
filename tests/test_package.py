
from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path


class TestPackage(TestCase):


	def test_import_pg_1(self):
		import pygem as pg
		params = pg.params.FFDParameters()


	def test_import_pg_2(self):
		import pygem as pg
		mat = pg.affine.angles2matrix(2)


	def test_import_pg_3(self):
		import pygem as pg
		fh = pg.filehandler.FileHandler()


	def test_import_pg_4(self):
		import pygem as pg
		unvh = pg.unvhandler.UnvHandler()


	def test_import_pg_5(self):
		import pygem as pg
		stlh = pg.stlhandler.StlHandler()


	def test_import_pg_6(self):
		import pygem as pg
		opfh = pg.openfhandler.OpenFoamHandler()


	def test_import_pg_7(self):
		import pygem as pg
		vtkh = pg.vtkhandler.VtkHandler()


	def test_import_pg_8(self):
		import pygem as pg
		igesh = pg.igeshandler.IgesHandler()
		
	
	def test_import_pg_9(self):
		import pygem as pg
		guih = pg.gui.Gui()
	

	def test_modules_name(self):
		# it checks that __all__ includes all the .py files in pygem folder
		import pygem
		package = pygem
		
		f_aux = []
		for (__, __, filenames) in walk('pygem'):
			f_aux.extend(filenames)

		f = []
		for i in f_aux:
			file_name, file_ext = path.splitext(i)
			if file_name != '__init__' and file_ext == '.py':
				f.append(file_name)
		
		self.assertItemsEqual(package.__all__, f)
