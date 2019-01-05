package edu.neu.ccs.pyramid.eval;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.ArgSort;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/20/15.
 */
public class NDCG {
    /**
     * all other ndcg methods use this one internally
     * @param gradesInRankedList true relevance grades in the ranked list
     * @param truncation
     * @return
     */
    public static double ndcg(double[] gradesInRankedList, int truncation){
        int truncationUsed = truncation;
        if (truncation > gradesInRankedList.length){
            System.out.println("Warning: truncation "+truncation+" is bigger than the list length "
                    +gradesInRankedList.length
                    +". Try to use the list length as truncation");
            truncationUsed = gradesInRankedList.length;
        }
        return dcg(gradesInRankedList,truncationUsed)/idcg(gradesInRankedList,truncationUsed);
    }

    /**
     * truncation = full list length
     * @param gradesInRandedList true relevance grades in the ranked list
     * @return
     */
    public static double ndcg(double[] gradesInRandedList){
        int truncation = gradesInRandedList.length;
        return ndcg(gradesInRandedList,truncation);
    }

    /**
     *
     * @param labels true grades, original order as in data set
     * @param scores prediction score of each doc, original order as in data set
     * @param truncation
     * @return
     */
    public static double ndcg(double[] labels, double[] scores, int truncation){
        if (labels.length!=scores.length){
            throw  new IllegalArgumentException("lengths are different");
        }
        int numData = labels.length;
        int[] sortedIndices = ArgSort.argSortDescending(scores);
        double[] gradesSortedWithScores = new double[numData];
        for (int i=0;i<numData;i++){
            gradesSortedWithScores[i] = labels[sortedIndices[i]];
        }
        return ndcg(gradesSortedWithScores,truncation);
    }

    /**
     * truncation = full list length
     * @param labels true grades, original order as in data set
     * @param scores prediction score of each doc, original order as in data set
     * @return
     */
    public static double ndcg(double[] labels, double[] scores){
        return ndcg(labels,scores,labels.length);
    }

    public static double instanceNDCG(MultiLabelClassifier.ClassProbEstimator classifier, MultiLabelClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().mapToDouble(i->{
            double[] binaryLabels = new double[classifier.getNumClasses()];
            MultiLabel multiLabel = dataSet.getMultiLabels()[i];
            for (int l:multiLabel.getMatchedLabels()) {
                binaryLabels[l] = 1;
            }
            double[] probs = classifier.predictClassProbs(dataSet.getRow(i));
            return ndcg(binaryLabels, probs);
        }).average().getAsDouble();
    }





    //==========PRIVATE==========

    // ideal dcg
    private static double idcg(double[] gradesInRankedList, int truncation){
        //should not sort the original one
        double[] sortedGrades = Arrays.copyOf(gradesInRankedList, gradesInRankedList.length);
        Arrays.sort(sortedGrades);
        ArrayUtils.reverse(sortedGrades);
        return dcg(sortedGrades,truncation);
    }

    /**
     * second formula in http://en.wikipedia.org/wiki/Discounted_cumulative_gain
     * use base 2 log
     * @param gradesInRankedList true relevance grades in the ranked list
     * @param truncation
     * @return
     */
    private static double dcg(double[] gradesInRankedList, int truncation){
        return IntStream.range(0, truncation).parallel()
                .mapToDouble(i-> {
                    double nominator = FastMath.pow(2, gradesInRankedList[i])-1;
                    //rank starts at 1
                    double denominator = FastMath.log(2,i + 2);
                    return nominator/denominator;
                }).sum();
    }
}
