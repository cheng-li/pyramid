package edu.neu.ccs.pyramid.core.eval;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;

/**
 * Created by chengli on 10/3/14.
 */
public class MacroAveragedMeasures {
    private double f1;

    /**
     * this is for single label dataset
     * @param confusionMatrix
     */
    public MacroAveragedMeasures(ConfusionMatrix confusionMatrix){
        int numClasses = confusionMatrix.getNumClasses();
        double sum = 0;
        for (int k=0;k<numClasses;k++){
            PerClassMeasures measures = new PerClassMeasures(confusionMatrix,k);
            sum += measures.getF1();
        }
        this.f1 = sum/numClasses;
    }

    public MacroAveragedMeasures(MultiLabelClassifier classifier, MultiLabelClfDataSet dataSet){
        int numClasses = dataSet.getNumClasses();
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        MultiLabel[] predictions = classifier.predict(dataSet);
        double sum = 0;
        for (int k=0;k<numClasses;k++){
            PerClassMeasures measures = new PerClassMeasures(multiLabels,predictions,k);
            sum += measures.getF1();
        }
        this.f1 = sum/numClasses;
    }



    public double getF1() {
        return f1;
    }

    @Override
    public String toString() {
        return "MacroAveragedMeasures{" +
                "f1=" + f1 +
                '}';
    }
}
