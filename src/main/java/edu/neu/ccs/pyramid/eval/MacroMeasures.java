package edu.neu.ccs.pyramid.eval;

/**
 * Created by Rainicy on 8/11/15.
 */
public class MacroMeasures extends LabelBasedMeasures{

    /**
     * Construct function: initialize each variables.
     *
     * @param numLabels
     */
    public MacroMeasures(int numLabels) {
        super(numLabels);
    }

    /**
     * Returns the average Macro precision for all labels.
     * @return average precision
     */
    public double getPrecision() {
        double sum = 0.0;
        for (int i=0; i<numLabels; i++) {
            sum += ConfusionMatrixMeasures.precision(truePositives[i],falsePositives[i],falseNegatives[i]);
        }
        return sum / numLabels;
    }

    /**
     * Returns the Macro precision by given label index.
     * @param labelIndex
     * @return precision of given label index
     */
    public double getPrecision(int labelIndex) {
        return ConfusionMatrixMeasures.precision(truePositives[labelIndex],
                falsePositives[labelIndex],falseNegatives[labelIndex]);
    }


    /**
     * Returns the average Macro recall for all labels.
     * @return average recall
     */
    public double getRecall() {
        double sum = 0.0;

        for (int i=0; i<numLabels; i++) {
            sum += ConfusionMatrixMeasures.recall(truePositives[i],
                    falsePositives[i],falseNegatives[i]);
        }

        return sum / numLabels;
    }

    /**
     * Returns the Macro recall by given label index.
     * @param labelIndex
     * @return recall of given label index
     */
    public double getRecall(int labelIndex) {
        return ConfusionMatrixMeasures.recall(truePositives[labelIndex],
                falsePositives[labelIndex],falseNegatives[labelIndex]);
    }


    /**
     * Returns the average Macro specificity for all labels.
     * @return average specificity
     */
    public double getSpecificity() {
        double sum = 0.0;

        for (int i=0; i<numLabels; i++) {
            sum += ConfusionMatrixMeasures.specificity(trueNegatives[i],falsePositives[i],falseNegatives[i]);
        }

        return sum / numLabels;
    }

    /**
     * Returns the Macro specificity by given label index.
     * @param labelIndex
     * @return specificity of given label index
     */
    public double getSpecificity(int labelIndex) {
        return ConfusionMatrixMeasures.specificity(trueNegatives[labelIndex],
                falsePositives[labelIndex],falseNegatives[labelIndex]);
    }

    /**
     * Returns the average Macro F score for all labels.
     * @param beta
     * @return average F score
     */
    public double getFScore(double beta) {
        double sum = 0.0;

        for (int i=0; i<numLabels; i++) {
            sum += ConfusionMatrixMeasures.fScore(truePositives[i],
                    falsePositives[i],falseNegatives[i],beta);
        }
        return sum / numLabels;
    }

    /**
     * Returns the Macro F score by given label index.
     * @param beta
     * @param labelIndex
     * @return F score of given label index
     */
    public double getFScore(double beta, int labelIndex) {
        return ConfusionMatrixMeasures.fScore(truePositives[labelIndex],
                falsePositives[labelIndex],falseNegatives[labelIndex],
                beta);
    }

    /**
     * Returns the average Macro F1 score for all labels.
     * @return average F1 score
     */
    public double getF1Score() {
        return getFScore(1.0);
    }

    /**
     * Returns the Macro F1 score by given label index.
     * @param labelIndex
     * @return F1 score of given label index
     */
    public double getF1Score(int labelIndex) {
        return getFScore(1.0, labelIndex);
    }

    @Override
    public String toString() {
        return "Macro-Precision:\t" + getPrecision() +
                "\nMacro-Recall:\t" + getRecall() +
                "\nMacro-Specificity:\t" + getSpecificity() +
                "\nMacro-F1Score:\t" + getF1Score();
    }
}

