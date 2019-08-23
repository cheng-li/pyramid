# Pyramid 

## A Java Machine Learning Library

Pyramid is a Java machine learning library which implements many state-of-the-art machine learning algorithms, including

* Binary and Multi-class classification algorithms:
    * Logistic Regression with L1 regularization (Lasso), L2 regularization (Ridge) and L1+L2 regularization (Elastic-net)
    * Variational Bayesian Logistic Regression
    * [Gradient Boosted Trees](https://github.com/cheng-li/pyramid/wiki/GB-Classifier)
    * Naive Bayes
    * Error-Correcting Output Codes (ECOC)
    * Support Vector Machines (SVM)
* Multi-label classification algorithms:
    * Binary Relevance
    * Power Set
    * Probabilistic Classifier Chain (PCC)
    * [Conditional Random Field (CRF)](https://github.com/cheng-li/pyramid/wiki/CRF)
    * [Multi-label Gradient Boosted Trees](https://github.com/cheng-li/pyramid/wiki/CBM)
    * [Conditional Bernoulli Mixture (CBM)](https://github.com/cheng-li/pyramid/wiki/CBM)
    * [BR-rerank](https://github.com/cheng-li/pyramid/wiki/BR-rerank)
* Regression algorithms:
    * Linear Regression with L1 regularization (Lasso), L2 regularization (Ridge) and L1+L2 regularization (Elastic-net)
    * Variational Bayesian Linear Regression
    * Regression Tree
    * [Gradient Boosted Trees](https://github.com/cheng-li/pyramid/wiki/GB-Regressor)
* Learning to rank algorithms:
    * LambdaMART
* Clustering: 
    * K Means
    * Gaussian Mixture
    * [Bernoulli Mixture](https://github.com/cheng-li/pyramid/wiki/Bernoulli-Mixtures)

_At the moment, not all algorithms are released. We are actively working on tidying up the source files and adding documentations. We will release a few algorithms at a time when they are ready and hope to have all algorthms released soon!_
## **Requirements**
If you just want to use pyramid as a command line tool (which is very simple), all you need is [Java 8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html).

If you are also a Java developer and wish to call Pyramid Java APIs, you will also need [Maven](https://maven.apache.org/).

## **Setup**
Pyramid doesn't require any installation effort. All you need is downloading the latest [pre-compiled package](https://github.com/cheng-li/pyramid/releases) (with a name like pyramid-x.x.x.zip) and decompressing it. Now you can move into the created folder and type 

`./pyramid config/welcome.properties`

You will see a welcome message and that means everything is working perfectly.

Windows users please see the [notes](https://github.com/cheng-li/pyramid/wiki/Notes-for-Windows-Users).
## **Command Line Usage**
All algorithms/functions implemented in Pyramid can be run though a simple command, with the following syntax:

`./pyramid <properties_file>`

Example: 

`./pyramid config/welcome.properties`

or

`./pyramid config/cbm.properties`

`pyramid` is a launcher script and `<properties_file>` is a file specifying the name of the algorithm and all necessary parameters, such as the input data, output folder, and learning algorithm hyper parameters. The `<properties_file>` can be specified by either an absolute or a relative path.

To run different algorithms, you just need to invoke the program with different properties files. The list of available algorithms and their corresponding properties file templates can be found in the [Wiki](https://github.com/cheng-li/pyramid/wiki#a-java-machine-learning-library).
 

## **Building from Source**
_If you are a Java developer who prefer working with the source code or want to contribute to the Pyramid package:_

Pyramid uses [Maven](https://maven.apache.org/) for its build system.

To compile and package the project from the source code, simply run the `mvn clean package -DskipTests` command in the cloned directory. The compressed package will be created under the core/target/releases directory.

## Feedback
We welcome your feedback on the package. To ask questions, request new features or report bugs, please contact Cheng Li  via chengli.email@gmail.com.

Answers to some commonly asked questions can be found in [FAQ](https://github.com/cheng-li/pyramid/wiki/FAQ).
