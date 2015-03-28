package edu.neu.ccs.pyramid.classification.dirty_naive_bayes;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 3/28/15.
 */
public class NBTrainer {
    public static NaiveBayes train(ClfDataSet dataSet){
        NaiveBayes naiveBayes = new NaiveBayes(dataSet.getNumClasses(),dataSet.getNumFeatures());
        int numClasses = dataSet.getNumClasses();
        int numDataPoints = dataSet.getNumDataPoints();
        int[] countPerClass = DataSetUtil.getCountPerClass(dataSet);
        for (int k=0;k<numClasses;k++){
            naiveBayes.priors[k] = ((double)countPerClass[k])/numDataPoints;
        }

        IntStream.range(0,dataSet.getNumFeatures()).parallel()
                .forEach(j -> updateOneColumn(naiveBayes, dataSet, countPerClass, j));
        for (int k=0;k<numClasses;k++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                naiveBayes.logPositive[k][j] = Math.log(naiveBayes.conditionals[k][j]);
                naiveBayes.logNegative[k][j] = Math.log(1-naiveBayes.conditionals[k][j]);
            }
        }


        return naiveBayes;

    }

    public static void updateOneColumn(NaiveBayes naiveBayes, ClfDataSet dataSet, int[] countPerClass, int featureIndex){
        double[] positiveCountPerClass = new double[dataSet.getNumClasses()];
        Vector vector = dataSet.getColumn(featureIndex);
        int[] labels = dataSet.getLabels();
        for (Vector.Element element: vector.nonZeroes()){
            int dataIndex = element.index();
            int label = labels[dataIndex];
            positiveCountPerClass[label] += 1;
        }
        for (int k=0;k<dataSet.getNumClasses();k++){
            naiveBayes.conditionals[k][featureIndex] = (positiveCountPerClass[k]+1)/countPerClass[k];
        }
    }
}
