package edu.neu.ccs.pyramid.eval;

/**
 * Created by chengli on 10/3/14.
 */
public class MacroAveragedMeasures {
    private double f1;

    public MacroAveragedMeasures(ConfusionMatrix confusionMatrix){
        int numClass = confusionMatrix.getNumClasses();
        double sum = 0;
        for (int k=0;k<numClass;k++){
            PerClassMeasures measures = new PerClassMeasures(confusionMatrix,k);
            sum += measures.getF1();
        }
        this.f1 = sum/numClass;
    }

    public double getF1() {
        return f1;
    }

    @Override
    public String toString() {
        return "MacroAveraged{" +
                "f1=" + f1 +
                '}';
    }
}
