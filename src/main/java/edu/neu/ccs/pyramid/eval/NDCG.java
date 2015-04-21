package edu.neu.ccs.pyramid.eval;

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
     * @param gradesInRandedList true relevance grades in the ranked list
     * @param truncation
     * @return
     */
    public static double ndcg(int[] gradesInRandedList, int truncation){
        int truncationUsed = truncation;
        if (truncation > gradesInRandedList.length){
            System.out.println("Warning: truncation "+truncation+" is bigger than the list length "
                    +gradesInRandedList.length
                    +". Try to use the list length as truncation");
            truncationUsed = gradesInRandedList.length;
        }
        return dcg(gradesInRandedList,truncationUsed)/idcg(gradesInRandedList,truncationUsed);
    }

    /**
     * truncation = full list length
     * @param gradesInRandedList true relevance grades in the ranked list
     * @return
     */
    public static double ndcg(int[] gradesInRandedList){
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
    public static double ndcg(int[] labels, double[] scores, int truncation){
        if (labels.length!=scores.length){
            throw  new IllegalArgumentException("lengths are different");
        }
        int numData = labels.length;
        int[] sortedIndices = ArgSort.argSortDescending(scores);
        int[] gradesSortedWithScores = new int[numData];
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
    public static double ndcg(int[] labels, double[] scores){
        return ndcg(labels,scores,labels.length);
    }







    //==========PRIVATE==========

    private static double idcg(int[] gradesInRandedList, int truncation){
        //should not sort the original one
        int[] sortedGrades = Arrays.copyOf(gradesInRandedList, gradesInRandedList.length);
        Arrays.sort(sortedGrades);
        ArrayUtils.reverse(sortedGrades);
        return dcg(sortedGrades,truncation);
    }

    /**
     * second formula in http://en.wikipedia.org/wiki/Discounted_cumulative_gain
     * use base 2 log
     * @param gradesInRandedList true relevance grades in the ranked list
     * @param truncation
     * @return
     */
    private static double dcg(int[] gradesInRandedList, int truncation){
        return IntStream.range(0, truncation).parallel()
                .mapToDouble(i-> {
                    double nominator = FastMath.pow(2, gradesInRandedList[i])-1;
                    //rank starts at 1
                    double denominator = FastMath.log(2,i + 2);
                    return nominator/denominator;
                }).sum();
    }
}
