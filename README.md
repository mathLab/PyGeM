# PyGeM [![Build Status](https://travis-ci.org/mathLab/PyGeM.svg)](https://travis-ci.org/mathLab/PyGeM) [![Coverage Status](https://coveralls.io/repos/github/mathLab/PyGeM/badge.svg?branch=master)](https://coveralls.io/github/mathLab/PyGeM?branch=master) [![Code Issues](https://www.quantifiedcode.com/api/v1/project/41f0acdbcba84e26a47ede5c57d62910/badge.svg)](https://www.quantifiedcode.com/app/project/41f0acdbcba84e26a47ede5c57d62910)
Python Geometrical Morphing.

![Python Geometrical Morphing](readme/logo_PyGeM_small.png)


## Description
**PyGeM** is a python library using **Free Form Deformation**, **Radial Basis Functions** and **Inverse Distance Weighting** to parametrize and morph complex geometries.  It is ideally suited for actual industrial problems, since it allows to handle:

- Computer Aided Design files (in .iges and .stl formats)
- Mesh files (in .unv and OpenFOAM formats)
- Output files (in .vtk format)
- LS-Dyna Keyword files (.k format)

By now, it has been used with meshes with up to 14 milions of cells. Try with more and more complicated input files! 
See the **Examples** section below to have an idea of the potential of this package.


## Graphical User Interface
**PyGeM** is now provided with a very basic Graphical User Interface (GUI) that, in Ubuntu environment, looks like the one depicted below. This feature can be easily used even by the pythonists beginners with not much effort. Up to now, PyGeM GUI works on linux and Mac OS X computers.

Pick the geometry, the parameters file, set the name of the output and decide whether dump the FFD lattices or not. Now just click on the `Run PyGeM` button and that is it. For a demonstration, see the [video tutorial on YouTube](https://youtu.be/iAjGEhXs_ys).

<p align="center">
<img src="readme/gui_PyGeM.png" alt>
</p>
<p align="center">
<em>PyGeM GUI: how it appears when it pops up.</em>
</p>


## Dependencies and installation
**PyGeM** requires `numpy`, `scipy` and `matplotlib`. They can be easily installed via `pip`. 
Moreover **PyGeM** depends on `OCC` and `vtk`. These requirements cannot be satisfied through `pip`.
Please see the table below for instructions on how to satisfy the requirements.

| Package | Version  | Comment                                                                    |
|---------|----------|----------------------------------------------------------------------------|
| OCC     | == 0.17  | See pythonocc.org or github.com.tpaviot/pythonocc-core for instructions or `conda install -c https://conda.anaconda.org/dlr-sc pythonocc-core==0.17` |
| vtk     | >= 5.0   | Simplest solution is `conda install vtk`                                   |


The official distribution is on GitHub, and you can clone the repository using

```bash
> git clone https://github.com/mathLab/PyGeM
```

To install the package just type:

```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```
Alternatively, a way to run the PyGeM library is to use our prebuilt and high-performance Docker images.
Docker containers are extremely lightweight, secure, and are based on open standards that run on all major Linux distributions, macOS and Microsoft Windows platforms.

Install Docker for your platform by following [these instructions](https://docs.docker.com/engine/getstarted/step_one/).
If using the Docker Toolbox (macOS versions < 10.10 or Windows versions < 10), make sure you run all commands inside the Docker Quickstart Terminal.

Now we will pull the docker.io/pygemdocker/pygem image from our cloud infrastructure:
```bash
>  docker pull docker.io/pygemdocker/pygem:latest
```
Docker will pull the latest tag of the image pygemdocker/pygem from docker.io. The download is around 3.246 GB. The  image is a great place to start experimenting with PyGeM and includes all dependencies already compiled for you.
Once the download is complete you can start PyGeM for the first time. Just run:
```bash
>  docker run -ti  pygemdocker/pygem:latest
```
To facilitate the devoloping, using the text editor,version control and other tools already installed on your computers,
it is possible to share files from the host into the container:

```bash
>  docker run -ti -v $(pwd):/home/PyGeM/shared  pygemdocker/pygem:latest
```
To allow the X11 forwarding in the container, on Linux system just run:

```bash
>  docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/home/PyGeM/shared  pygemdocker/pygem:latest
```

For Windows system, you need to install Cygwin/X version and running the command in Cygwin terminal. While for mac system, you need to install xquartz. 

## Documentation
**PyGeM** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. You can view the documentation online [here](http://mathlab.github.io/PyGeM/). To build the html versions of the docs locally simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse.


## Testing
We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/PyGeM).

To run tests locally:

```bash
> python test.py
```


## Authors and contributors
**PyGeM** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Marco Tezzele](mailto:marcotez@gmail.com)
* [Nicola Demo](mailto:demo.nicola@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it). We thank [Filippo Salmoiraghi](mailto:filippo.salmoiraghi@gmail.com) for the original idea behind this package and the major contributions.

Contact us by email for further information or questions about **PyGeM**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## Examples
You can find useful tutorials on how to use the package in the `tutorials` folder.
Here we show three applications, taken from the **naval**, **nautical** and **automotive** engineering fields. On the other hand, the provided tutorials are related to easier geometries.
<p align="center">
<img src="readme/DTMB_ffd.png" alt>
</p>
<p align="center">
<em>DTMB-5415 hull: morphing of the bulbous bow starting from an industrial .iges CAD file.</em>
</p>

<p align="center">
<img src="readme/scafoYZshift.gif" alt>
</p>
<p align="center">
<em>MCY hull: morphing of the exhaust gasses devices starting from an industrial .stl file.</em>
</p>

<p align="center">
<img src="readme/drivAer_ffd.png" alt>
</p>
<p align="center">
<em>DrivAer model: morphing of the bumper starting from an OpenFOAM mesh file.</em>
</p>


## How to contribute
We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use a tab of 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
