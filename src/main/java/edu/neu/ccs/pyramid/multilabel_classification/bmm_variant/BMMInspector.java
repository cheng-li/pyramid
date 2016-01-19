package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Matrices;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.ojalgo.access.Access2D;
import org.ojalgo.matrix.BasicMatrix;
import org.ojalgo.matrix.PrimitiveMatrix;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 12/25/15.
 */
public class BMMInspector {
    private static BasicMatrix.Factory<PrimitiveMatrix> factory = PrimitiveMatrix.FACTORY;

    public static Weights getMean(BMMClassifier bmm, int label){
        int numClusters = bmm.getNumClusters();
        int length = ((LogisticRegression)bmm.getBinaryClassifiers()[0][0]).getWeights().getAllWeights().size();
        int numFeatures = ((LogisticRegression)bmm.getBinaryClassifiers()[0][0]).getNumFeatures();
        Vector mean = new DenseVector(length);
        for (int k=0;k<numClusters;k++){
            mean = mean.plus(((LogisticRegression)bmm.getBinaryClassifiers()[k][label]).getWeights().getAllWeights());
        }

        mean = mean.divide(numClusters);
        return new Weights(2,numFeatures,mean);
    }


    public static double distanceFromMean(BMMClassifier bmm){
        int numClasses = bmm.getNumClasses();
        return IntStream.range(0,numClasses).mapToDouble(l -> distanceFromMean(bmm,l)).average().getAsDouble();
    }

    public static double distanceFromMean(BMMClassifier bmm, int label){
        Classifier.ProbabilityEstimator[][] logistics = bmm.getBinaryClassifiers();
        int numClusters = bmm.getNumClusters();
        int numFeatures =  ((LogisticRegression)logistics[0][0]).getNumFeatures();

        Vector positiveAverageVector = new DenseVector(numFeatures);
        for (int k=0;k<numClusters;k++){
            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            positiveAverageVector = positiveAverageVector.plus(positiveVector);
        }


        positiveAverageVector = positiveAverageVector.divide(numClusters);

        double dis = 0;
        for (int k=0;k<numClusters;k++){

            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            dis += positiveVector.minus(positiveAverageVector).norm(2);
        }
        return dis/numClusters;
    }

    public static List<Map<MultiLabel,Double>> visualizeClusters(BMMClassifier bmm, MultiLabelClfDataSet dataSet){
        int numClusters = bmm.getNumClusters();
        List<Map<MultiLabel,Double>> list = new ArrayList<>();
        for (int k=0;k<numClusters;k++){
            list.add(new HashMap<>());
        }

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            double[] clusterProbs = bmm.getMultiClassClassifier().predictClassProbs(dataSet.getRow(i));
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int k=0;k<numClusters;k++){
                Map<MultiLabel,Double> map = list.get(k);
                double count = map.getOrDefault(multiLabel,0.0);
                double newcount = count+clusterProbs[k];
                map.put(multiLabel,newcount);
            }
        }
        return list;
    }



    public static void visualizePrediction(BMMClassifier bmmClassifier, Vector vector, LabelTranslator labelTranslator){
        int numClusters = bmmClassifier.getNumClusters();
        int numClasses = bmmClassifier.getNumClasses();
        double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(vector);
        double[][] probabilities = new double[numClusters][numClasses];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numClasses;l++){
                probabilities[k][l]=bmmClassifier.getBinaryClassifiers()[k][l].predictClassProb(vector,1);
            }
        }

        int[] sorted = ArgSort.argSortDescending(proportions);
        double[] topProbs = probabilities[sorted[0]];
        MultiLabel trivalPred = new MultiLabel();
        for (int l=0;l<numClasses;l++){
            if (topProbs[l]>=0.5){
                trivalPred.addLabel(l);
            }
        }

        MultiLabel secondPred = new MultiLabel();
        for (int l=0;l<numClasses;l++){
            if (probabilities[sorted[1]][l]>=0.5){
                secondPred.addLabel(l);
            }
        }

        MultiLabel predicted = bmmClassifier.predict(vector);
        if (!predicted.equals(trivalPred)){
            System.out.println("interesting case !");
            if (!predicted.equals(secondPred)){
                System.out.println("very interesting case !");
            }
        }

//        System.out.println("proportion = "+ Arrays.toString(proportions));

        System.out.println("perplexity="+ Math.pow(2,Entropy.entropy2Based(proportions)));
        double[] sortedPorportions = new double[numClusters];
        for (int t=0;t<sorted.length;t++){
            int k = sorted[t];
//            System.out.println("rank "+t);
//            System.out.println("cluster "+k);
//            System.out.println(proportions[k]);
            StringBuilder sb = new StringBuilder();
//            sb.append("[");
//            for (int l=0;l<numClasses;l++){
//                double p = probabilities[k][l];
//                if (p>0.01){
//                    sb.append(l).append(":").append(p).append(", ");
//                }
//
//            }
//            sb.append("]");
//            System.out.println(sb.toString());
            System.out.println("prob"+(t+1)+" = " +Arrays.toString(probabilities[k]));
            sortedPorportions[t] = proportions[k];
        }
        for (int t=0;t<sorted.length;t++){
            int k = sorted[t];
            double[] probs = probabilities[k];
            List<String> labels = new ArrayList<>();
            for (int l=0;l<numClasses;l++){
                if (probs[l]>0.5){
                    labels.add("\""+labelTranslator.toExtLabel(l)+"\"");
                }
            }
            System.out.println("labels"+(t+1)+" = "+labels);
        }
        System.out.println("proportions = "+Arrays.toString(sortedPorportions));
    }

    public static void covariance(BMMClassifier bmmClassifier, Vector vector, LabelTranslator labelTranslator){
        int numClusters = bmmClassifier.getNumClusters();
        int numClasses = bmmClassifier.getNumClasses();
        double[] proportions = bmmClassifier.getMultiClassClassifier().predictClassProbs(vector);
        double[][] probabilities = new double[numClusters][numClasses];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numClasses;l++){
                probabilities[k][l]=bmmClassifier.getBinaryClassifiers()[k][l].predictClassProb(vector,1);
            }
        }
        // column vector
        Access2D.Builder<PrimitiveMatrix> meanBuilder = factory.getBuilder(numClasses,1);
        for (int l=0;l<numClasses;l++){
            double sum = 0;
            for (int k=0;k<numClusters;k++){
                sum += proportions[k]*probabilities[k][l];
            }
            meanBuilder.set(l,0,sum);
        }
        BasicMatrix mean = meanBuilder.build();
//        System.out.println(mean);

        List<BasicMatrix> mus  = new ArrayList<>();
        for (int k=0;k<numClusters;k++){
            Access2D.Builder<PrimitiveMatrix> muBuilder = factory.getBuilder(numClasses,1);
            for (int l=0;l<numClasses;l++){
                muBuilder.set(l,0,probabilities[k][l]);
            }
            BasicMatrix muK = muBuilder.build();
            mus.add(muK);
        }

        List<BasicMatrix> sigmas = new ArrayList<>();
        for (int k=0;k<numClusters;k++){
            Access2D.Builder<PrimitiveMatrix> sigmaBuilder = factory.getBuilder(numClasses,numClasses);
            for (int l=0;l<numClasses;l++){
                double v= probabilities[k][l]*(1-probabilities[k][l]);
                sigmaBuilder.set(l,l,v);
            }
            BasicMatrix sigmaK = sigmaBuilder.build();
            sigmas.add(sigmaK);
        }

        BasicMatrix covariance = factory.makeZero(numClasses,numClasses);
        for (int k=0;k<numClusters;k++){
            BasicMatrix muk = mus.get(k);
            BasicMatrix toadd = (sigmas.get(k).add(muk.multiply(muk.transpose()))).multiply(proportions[k]);
            covariance = covariance.add(toadd);
        }
        covariance = covariance.subtract(mean.multiply(mean.transpose()));

//        System.out.println("covariance = "+ Matrices.display(covariance));

        Access2D.Builder<PrimitiveMatrix> correlationBuilder = factory.getBuilder(numClasses,numClasses);

        for (int l=0;l<numClasses;l++){
            for (int j=0;j<numClasses;j++){
                double v = covariance.get(l,j).doubleValue()/(Math.sqrt(covariance.get(l,l).doubleValue())*Math.sqrt(covariance.get(j,j).doubleValue()));
                correlationBuilder.set(l,j,v);
            }
        }

        BasicMatrix correlation = correlationBuilder.build();
//        System.out.println("correlation = "+ Matrices.display(correlation));

        List<Pair<String,Double>> list = new ArrayList<>();
        for (int l=0;l<numClasses;l++){
            for (int j=0;j<l;j++){
                String s = ""+labelTranslator.toExtLabel(l)+", "+labelTranslator.toExtLabel(j);
                double v = correlation.get(l,j).doubleValue();
                Pair<String,Double> pair = new Pair<>(s,v);
                list.add(pair);
            }
        }

        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<Pair<String,Double>> top = list.stream().sorted(comparator.reversed()).limit(20).collect(Collectors.toList());
        System.out.println(top);

    }
}
