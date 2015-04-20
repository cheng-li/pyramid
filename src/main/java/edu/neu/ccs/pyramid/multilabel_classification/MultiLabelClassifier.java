package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/27/14.
 */
public interface MultiLabelClassifier extends Serializable{
    int  getNumClasses();
    MultiLabel predict(Vector vector);
    default List<MultiLabel> predict(MultiLabelClfDataSet dataSet){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToObj(i -> predict(dataSet.getRow(i)))
                .collect(Collectors.toList());
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

    public interface ClassScoreEstimator extends MultiLabelClassifier{
        public double predictClassScore(Vector vector, int k);
        public default double[] predictClassScores(Vector vector){
            return IntStream.range(0,getNumClasses()).mapToDouble(k -> predictClassScore(vector,k))
                    .toArray();

        }
    }



    public interface ClassProbEstimator extends MultiLabelClassifier{
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

    public interface AssignmentProbEstimator extends MultiLabelClassifier{
        public double predictAssignmentProb(Vector vector, MultiLabel assignment);
    }

}
