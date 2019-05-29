The scripts in this directory are written in Python 2,
and require numpy and matplotlib to run.

The code algorithmic code is in kmeans.py and pca.py.
The remaining scripts are for generating plots and data.

    python2 kmeans.py k

        Prints SSE for k-means with k=k.
        First column: k. Second column: iteration. Third column: SSE.

    bash q2.sh

        Runs kmeans.py with different parameters.
        Requires GNU parallel to be installed.

    python2 q2.1-plot.py
    python2 q2.2-plot.py

        Reads kmeans.data and generates SSE plots from it.

    python2 pca_1.py

        Prints top 10 eigenvalues.

    python2 pca_2.py

        Creates a plot of the mean image and the top 10 eigenvectors.

    python2 pca_3.py

        Creates a plot of the most representative images for each of the top 10 eigenvectors.
