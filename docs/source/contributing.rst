How to contribute
===================

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

Development Setup
-----------------

First, install PyGeM in development mode with all development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This installs the package in editable mode with all development extras (testing, documentation, tutorials, and code quality tools).

Common Development Tasks
------------------------

**Running tests:**

.. code-block:: bash
   
   pip install -e ".[dev]"      # or ".[test]" if just running tests
   pytest

**Building documentation:**

.. code-block:: bash

   pip install ".[docs]"
   cd docs
   make html

**Code formatting:**

.. code-block:: bash

   black .

**Code linting:**

.. code-block:: bash

   pylint pygem

**Running tutorials:**

.. code-block:: bash

   pip install ".[tut]"

Submitting a patch:
-------------------

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of forking the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use a tab of 4
     spaces to indent the code. You should also run pylint over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using coveralls that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have well-formed commit messages for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a pull request. Please,
     remember to rebase properly in order to maintain a clean, linear git history.