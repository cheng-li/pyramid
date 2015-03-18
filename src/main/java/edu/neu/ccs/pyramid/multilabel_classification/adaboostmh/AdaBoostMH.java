package edu.neu.ccs.pyramid.multilabel_classification.adaboostmh;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.Vector;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 3/15/15.
 */
public class AdaBoostMH implements MultiLabelClassifier {
    private static final long serialVersionUID = 2L;
    private List<List<Regressor>> regressors;
    private int numClasses;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    public AdaBoostMH(int numClasses) {
        this.numClasses = numClasses;
        this.regressors = new ArrayList<>(this.numClasses);
        for (int k=0;k<this.numClasses;k++){
            List<Regressor> regressorsClassK  = new ArrayList<>();
            this.regressors.add(regressorsClassK);
        }
    }

    public int getNumClasses() {
        return numClasses;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel prediction = new MultiLabel();
        for (int k=0;k<numClasses;k++){
            double score = this.predictClassScore(vector, k);
            if (score > 0){
                prediction.addLabel(k);
            }
        }
        return prediction;
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    public double predictClassScore(Vector vector, int k){
        List<Regressor> regressorsClassK = this.regressors.get(k);
        double score = 0;
        for (Regressor regressor: regressorsClassK){
            score += regressor.predict(vector);
        }
        return score;
    }

    void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }

    void addRegressor(Regressor regressor, int k){
        this.regressors.get(k).add(regressor);
    }


    public List<Regressor> getRegressors(int k){
        return this.regressors.get(k);
    }


    double[] calClassScores(Vector vector){
        int numClasses = this.numClasses;
        double[] scores = new double[numClasses];
        for (int k=0;k<numClasses;k++){
            scores[k] = this.predictClassScore(vector, k);
        }
        return scores;
    }

    public static AdaBoostMH deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }

    /**
     * de-serialize from file
     * @param file
     * @return
     * @throws Exception
     */
    public static AdaBoostMH deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            AdaBoostMH boosting = (AdaBoostMH)objectInputStream.readObject();
            return boosting;
        }
    }

}
