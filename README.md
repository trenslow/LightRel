# LightRel

This is the system used to participate in Task 7 of the International Workshop on Semantic Evaluation 2018.

TODO:

- add instructions for installing and running the precursor to this system, which was developed on the dataset from  Task 8 of the International Workshop on Semantic Evaluation 2010

## System Requirements

LightRel was developed and tested on Python 3.6. LibLinear 2.11 must be installed somewhere on your system in order to train models and predict on test data. It can be downloaded [here](https://www.csie.ntu.edu.tw/~cjlin/liblinear/#download). Don't forget to cite them if you use LightRel!

## Installation

To download LightRel, simply clone this repository:
```
git clone https://github.com/trenslow/LightRel
```

After downloading, change into the system directory:

```
cd SemEval18task7/
```

Here you will find the system script lightRel.sh. Make sure to edit this script so that it points to the location of LibLinear on your system. You can find more about running the system and changing the LibLinear parameters below.

LightRel also requires an embedding file, which can be downloaded [here](https://cloud.dfki.de/owncloud/index.php/s/WKOCMj5UYiSVZeR). The README file at that link goes into more depth about how the embeddings were created.

There are two files to choose from at this link (both have .wcs in their names); the one used in the competition is made from the dblp corpus. Once downloaded, unzip the embedding file into the features directory of the system with the following commands (with your own path to LightRel directory, of course):

```
cd ~/Downloads/
gunzip abstracts-dblp-semeval2018.wcs.txt.gz 
mv abstracts-dblp-semeval2018.wcs.txt ~/your/path/to/LightRel/SemEval18task7/features/
```

Once the paths are properly set and the embeddings are in the correct folder, LightRel can be run.

## Running System

To run the system, run the following in the SemEval18task7 directory:

```
./lightRel.sh k
```

An argument *k* must be included. If *k* is 0, then LightRel will perform competition mode, which trains on the data provided by the task organizers and then predicts on the test data.
For *k* greater than 0, it will run *k*-fold cross-validation. The results of each mode will be printed to the console. The system does not yet catch errors in the argument input, so anything other than a positive integer will produce unwanted results!

In the LightRel script, the parameters for LibLinear can be tuned as needed (don't forget to change the variable 'libLinDir' so that it points to the LibLinear directory on your system!).

In the parameters.py script, the different features can be turned on and off by setting the 'fire' variables to `True` or `False`, respectively. You can also choose the SemEval subtask by changing `task_number` variable to either '1.1' or '1.2'. All parameters are set to produce the best competition result for subtask 1.1 by default.

## Attribution

If you like LightRel and use it, we ask that you please cite it in your work.

> Tyler Renslow and Günter Neumann (2018). "LightRel at SemEval-2018 Task 7: Lightweight, Fast and Robust Relation Classification." In proceedings of the 12th International Workshop on Semantic Evaluation (SemEval-2018).
