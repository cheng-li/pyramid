package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.classification.logistic_regression.Weights;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Entropy;
import edu.neu.ccs.pyramid.util.ArgSort;
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
public class CBMInspector {
    private static BasicMatrix.Factory<PrimitiveMatrix> factory = PrimitiveMatrix.FACTORY;

    public static Set<Integer> usedFeatures(CBM cbm){
        Set<Integer> all = new HashSet<>();
        if (cbm.getMultiClassClassifier() instanceof LogisticRegression){
            all.addAll(LogisticRegressionInspector.usedFeaturesCombined((LogisticRegression)cbm.getMultiClassClassifier()));
        }

        for (int k=0;k<cbm.getNumComponents();k++){
            for (int l=0;l<cbm.getNumClasses();l++){
                if (cbm.getBinaryClassifiers()[k][l] instanceof LogisticRegression){
                    all.addAll(LogisticRegressionInspector.usedFeaturesCombined((LogisticRegression)cbm.getBinaryClassifiers()[k][l]));
                }

            }
        }
        return all;
    }


    public static int[] usedFeaturesByEachLabel(CBM cbm){
        int[] count = new int[cbm.getNumClasses()];
        for (int l=0;l<cbm.getNumClasses();l++) {
            Set<Integer> used = new HashSet<>();
            for (int k = 0; k < cbm.getNumComponents(); k++) {
                if (cbm.getBinaryClassifiers()[k][l] instanceof LogisticRegression){
                    used.addAll(LogisticRegressionInspector.usedFeaturesCombined((LogisticRegression)cbm.getBinaryClassifiers()[k][l]));
                }
            }
            count[l] = used.size();
        }
        return count;
    }



    public static String topLabels(CBM cbm, Vector vector, double probabilityThreshold){
        double[] marginals = cbm.predictClassProbs(vector);
        List<Pair<Integer, Double>> list = new ArrayList<>();
        Comparator<Pair<Integer, Double>> comparator = Comparator.comparing(Pair::getSecond);
        for (int l=0;l<cbm.getNumClasses();l++){
            list.add(new Pair<>(l, marginals[l]));
        }

        List<Pair<Integer, Double>> sorted = list.stream().filter(pair->pair.getSecond()>=probabilityThreshold)
                .sorted(comparator.reversed()).collect(Collectors.toList());
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<sorted.size();i++){
            Pair<Integer, Double> pair = sorted.get(i);
            sb.append(pair.getFirst()).append(":").append(pair.getSecond());
            if (i!=sorted.size()-1){
                sb.append(", ");
            }
        }
        return sb.toString();
    }

    public static Weights getMean(CBM bmm, int label){
        int numClusters = bmm.getNumComponents();
        int length = ((LogisticRegression)bmm.getBinaryClassifiers()[0][0]).getWeights().getAllWeights().size();
        int numFeatures = ((LogisticRegression)bmm.getBinaryClassifiers()[0][0]).getNumFeatures();
        Vector mean = new DenseVector(length);
        for (int k=0;k<numClusters;k++){
            mean = mean.plus(((LogisticRegression)bmm.getBinaryClassifiers()[k][label]).getWeights().getAllWeights());
        }

        mean = mean.divide(numClusters);
        return new Weights(2,numFeatures,mean);
    }


    public static double distanceFromMean(CBM bmm){
        int numClasses = bmm.getNumClasses();
        return IntStream.range(0,numClasses).mapToDouble(l -> distanceFromMean(bmm,l)).average().getAsDouble();
    }

    public static double distanceFromMean(CBM bmm, int label){
        Classifier.ProbabilityEstimator[][] logistics = bmm.getBinaryClassifiers();
        int numClusters = bmm.getNumComponents();
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

    public static List<Map<MultiLabel,Double>> visualizeClusters(CBM bmm, MultiLabelClfDataSet dataSet){
        int numClusters = bmm.getNumComponents();
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



    public static void visualizePrediction(CBM CBM, Vector vector, LabelTranslator labelTranslator){
        int numClusters = CBM.getNumComponents();
        int numClasses = CBM.getNumClasses();
        double[] proportions = CBM.getMultiClassClassifier().predictClassProbs(vector);
        double[][] probabilities = new double[numClusters][numClasses];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numClasses;l++){
                probabilities[k][l]= CBM.getBinaryClassifiers()[k][l].predictClassProb(vector,1);
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

        MultiLabel predicted = CBM.predict(vector);
        if (!predicted.equals(trivalPred)){
            System.out.println("interesting case !");
            if (!predicted.equals(secondPred)){
                System.out.println("very interesting case !");
            }
        }

//        System.out.println("proportion = "+ Arrays.toString(proportions));


        double[] sortedPorportions = new double[numClusters];
        for (int t=0;t<sorted.length;t++){
            int k = sorted[t];
            sortedPorportions[t] = proportions[k];
        }
        System.out.println("proportions = "+Arrays.toString(sortedPorportions));

        for (int t=0;t<sorted.length;t++){
            int k = sorted[t];
            System.out.println("prob"+(t+1)+" = " +Arrays.toString(probabilities[k]));
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

        System.out.println("perplexity="+ Math.pow(2,Entropy.entropy2Based(proportions)));

        for (int t=0;t<numClusters;t++){
            System.out.println("cluster"+(t+1)+" = " +Arrays.toString(probabilities[t]));
        }

    }

    public static void covariance(CBM CBM, Vector vector, LabelTranslator labelTranslator){
        int numClusters = CBM.getNumComponents();
        int numClasses = CBM.getNumClasses();
        double[] proportions = CBM.getMultiClassClassifier().predictClassProbs(vector);
        double[][] probabilities = new double[numClusters][numClasses];
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numClasses;l++){
                probabilities[k][l]= CBM.getBinaryClassifiers()[k][l].predictClassProb(vector,1);
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
