package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.Utils;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * mean average precision
 * Created by chengli on 12/27/16.
 */
public class MAP {
    /**
     * compute mean average precision over given labels
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double map(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet, List<Integer> labels){
        if (classifier.getNumClasses()!=dataSet.getNumClasses()){
            throw new IllegalArgumentException("classifier.getNumClasses()!=dataSet.getNumClasses()");
        }
        int numData = dataSet.getNumDataPoints();
        double[][] probs = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];

        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->probs[i] = classifier.predictClassProbs(dataSet.getRow(i)));

        double sum = 0;

        for (int l: labels){
            int[] binaryLabels = new int[numData];
            double[] marginals = new double[numData];
            for (int i=0;i<numData;i++){
                if (dataSet.getMultiLabels()[i].matchClass(l)){
                    binaryLabels[i] = 1;
                }
                marginals[i] = probs[i][l];
            }

            double averagePrecision = AveragePrecision.averagePrecision(binaryLabels, marginals);
//            System.out.println("AP for label "+l+"="+averagePrecision);
            sum += averagePrecision;
        }
        return sum/labels.size();
    }

    /**
     * label MAP; marginals are provided by the classifiers
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double map(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet){
        List<Integer> labels = IntStream.range(0, dataSet.getNumClasses()).boxed().collect(Collectors.toList());
        return map(classifier, dataSet, labels);
    }


    public static double[] labelMapDetail(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet){
        if (classifier.getNumClasses()!=dataSet.getNumClasses()){
            throw new IllegalArgumentException("classifier.getNumClasses()!=dataSet.getNumClasses()");
        }
        int numData = dataSet.getNumDataPoints();
        double[][] probs = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];

        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->probs[i] = classifier.predictClassProbs(dataSet.getRow(i)));
        double[] ap = new double[classifier.getNumClasses()];

        for (int l=0;l<classifier.getNumClasses();l++){
            int[] binaryLabels = new int[numData];
            double[] marginals = new double[numData];
            for (int i=0;i<numData;i++){
                if (dataSet.getMultiLabels()[i].matchClass(l)){
                    binaryLabels[i] = 1;
                }
                marginals[i] = probs[i][l];
            }

            ap[l] = AveragePrecision.averagePrecision(binaryLabels, marginals);
        }
        return ap;
    }


    /**
     * label MAP
     * marginals are estimated through support combinations
     * @param classifier
     * @param dataSet
     * @return
     */
    public static double mapBySupport(MultiLabelClassifier.AssignmentProbEstimator classifier, MultiLabelClfDataSet dataSet, List<MultiLabel> combinations){
        if (classifier.getNumClasses()!=dataSet.getNumClasses()){
            throw new IllegalArgumentException("classifier.getNumClasses()!=dataSet.getNumClasses()");
        }
        int numData = dataSet.getNumDataPoints();
        double[][] probs = new double[dataSet.getNumDataPoints()][dataSet.getNumClasses()];

        IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .forEach(i->{
                    double[] comProbs = classifier.predictAssignmentProbs(dataSet.getRow(i),combinations);
                    probs[i] = Utils.marginals(combinations, comProbs, classifier.getNumClasses());
                });

        double sum = 0;

        for (int l=0;l<dataSet.getNumClasses();l++){
            int[] binaryLabels = new int[numData];
            double[] marginals = new double[numData];
            for (int i=0;i<numData;i++){
                if (dataSet.getMultiLabels()[i].matchClass(l)){
                    binaryLabels[i] = 1;
                }
                marginals[i] = probs[i][l];
            }

            double averagePrecision = AveragePrecision.averagePrecision(binaryLabels, marginals);

            System.out.println("AP for label "+l+"="+averagePrecision);
            sum += averagePrecision;
        }
        return sum/dataSet.getNumClasses();
    }

    public static double instanceMAP(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().mapToDouble(i->{
            int[] binaryLabels = new int[classifier.getNumClasses()];
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l:multiLabel.getMatchedLabels()) {
                binaryLabels[l] = 1;
            }
            double[] probs = classifier.predictClassProbs(dataSet.getRow(i));
            return AveragePrecision.averagePrecision(binaryLabels, probs);
        }).average().getAsDouble();
    }

    public static double instanceMAP(MultiLabelClassifier.AssignmentProbEstimator classifier, MultiLabelClfDataSet dataSet, List<MultiLabel> combinations){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().mapToDouble(i->{
            int[] binaryLabels = new int[classifier.getNumClasses()];
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l:multiLabel.getMatchedLabels()) {
                binaryLabels[l] = 1;
            }
            double[] comProbs = classifier.predictAssignmentProbs(dataSet.getRow(i),combinations);
            double[] probs = Utils.marginals(combinations, comProbs, classifier.getNumClasses());
            return AveragePrecision.averagePrecision(binaryLabels, probs);
        }).average().getAsDouble();
    }


    public static double instanceMAP(double[][] marginals, MultiLabelClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().mapToDouble(i->{
            int[] binaryLabels = new int[dataSet.getNumClasses()];
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l:multiLabel.getMatchedLabels()) {
                binaryLabels[l] = 1;
            }
            return AveragePrecision.averagePrecision(binaryLabels, marginals[i]);
        }).average().getAsDouble();
    }


}
