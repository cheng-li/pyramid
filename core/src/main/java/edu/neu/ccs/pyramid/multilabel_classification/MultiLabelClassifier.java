package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public interface MultiLabelClassifier extends Serializable{
    int  getNumClasses();
    MultiLabel predict(Vector vector);
    default MultiLabel[] predict(MultiLabelClfDataSet dataSet){

        List<MultiLabel> results = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> predict(dataSet.getRow(i)))
                .collect(Collectors.toList());
        return results.toArray(new MultiLabel[results.size()]);
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

    interface ClassScoreEstimator extends MultiLabelClassifier{
        double predictClassScore(Vector vector, int k);
        default double[] predictClassScores(Vector vector){
            return IntStream.range(0,getNumClasses()).mapToDouble(k -> predictClassScore(vector,k))
                    .toArray();

        }
    }



    interface ClassProbEstimator extends MultiLabelClassifier{
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
    }

    interface AssignmentProbEstimator extends MultiLabelClassifier{
        double predictLogAssignmentProb(Vector vector, MultiLabel assignment);
        default double predictAssignmentProb(Vector vector, MultiLabel assignment){
            return Math.exp(predictLogAssignmentProb(vector, assignment));
        }

        /**
         * batch version
         * can be implemented more efficiently in individual classifiers
         * @param vector
         * @param assignments
         * @return
         */
        default double[] predictAssignmentProbs(Vector vector, List<MultiLabel> assignments){
            return Arrays.stream(predictLogAssignmentProbs(vector, assignments)).map(Math::exp).toArray();
        }


        default double[] predictLogAssignmentProbs(Vector vector, List<MultiLabel> assignments){
            double[] logProbs = new double[assignments.size()];
            for (int c=0;c<assignments.size();c++){
                logProbs[c] = predictLogAssignmentProb(vector, assignments.get(c));
            }
            return logProbs;
        }
    }

}
