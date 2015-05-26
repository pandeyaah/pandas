.. _style:

.. currentmodule:: pandas

.. ipython:: python
   :suppress:

   import numpy as np
   import pandas as pd
   np.set_printoptions(precision=4, suppress=True)
   pd.options.display.max_rows = 15

*******
Styling
*******

.. versionadded:: 0.17.1


You can apply **conditional formatting**, the visual styling of a DataFrame
depending on the data within, by using the ``.style`` property.
This is a property on every object that returns a ``Styler`` object, which has
useful methods for formatting and displaying DataFrames.

See the notebook for more.
