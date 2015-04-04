package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;


import java.io.*;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/13/14.
 */
public interface Classifier extends Serializable{
    int predict(Vector vector);

    int  getNumClasses();

    default int[] predict(ClfDataSet dataSet){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel().
                map(i -> predict(dataSet.getRow(i))).toArray();
    }

    default void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    default void serialize(String file) throws Exception{
        serialize(new File(file));
    }

    FeatureList getFeatureList();

    LabelTranslator getLabelTranslator();

    interface ScoreEstimator extends Classifier{
        public double predictClassScore(Vector vector, int k);
    }


    /**
     * Created by chengli on 8/14/14.
     */
    interface ProbabilityEstimator extends Classifier{
        double[] predictClassProbs(Vector vector);

        /**
         * in some cases, this can be implemented more efficiently
         * @param vector
         * @param classIndex
         * @return
         */
        default double predictClassProb(Vector vector, int classIndex){
            return predictClassProbs(vector)[classIndex];
        }

        default List<double[]> predictClassProbs(DataSet dataSet){
            return IntStream.range(0,dataSet.getNumDataPoints())
                    .parallel().mapToObj(i -> predictClassProbs(dataSet.getRow(i)))
                    .collect(Collectors.toList());
        }

        /**
         * by default, probabilities can be used for classification.
         * classifier should override this method if
         * calculation of probabilities is not necessary for classification, or
         * calculation of probabilities is too slow, or
         * calculation of probabilities is unstable
         * @param vector
         * @return
         */
        @Override
        default int predict(Vector vector){
            Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
            double[] probs = predictClassProbs(vector);
            return IntStream.range(0,probs.length)
                    .mapToObj(i-> new Pair<>(i,probs[i]))
                    .max(comparator).get().getFirst();
        }

    }
}