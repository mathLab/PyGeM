from setuptools import setup

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='pygem',
	  version='0.1',
	  description='Tools to apply FFD.',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 3 - Alpha',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='dimension reduction mathematics ffd',
	  url='https://github.com/mathLab/PyGeM',
	  authors='Filippo Salmoiraghi, Marco Tezzele',
	  author_email='filippo.salmoiraghi@gmail.com, marcotez@gmail.com',
	  license='MIT',
	  packages=['pygem'],
	  install_requires=[
	  		'numpy',
	  		'numpy-stl',
	  		'scipy',
	  		'matplotlib',
	  		'enum34'
	  ],
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  include_package_data=True,
	  zip_safe=False)
