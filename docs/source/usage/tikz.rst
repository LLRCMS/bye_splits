Producing ``tikz`` Standalone Pictures
***************************************

For the purpose of illustration, ``tikz`` standalone script have been included under ``docs/tikz/``.
To run them (taking ``docs/tikz/flowchart.tex`` as an example):

.. code-block:: shell
				
    cd docs/tikz/
    pdflatex -shell-escape flowchart.tex

The above should produce the ``flowchart.svg`` file.
The code depends on ``latex`` and ``pdf2svg``.
