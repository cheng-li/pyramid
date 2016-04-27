package edu.neu.ccs.pyramid.multilabel_classification.imlgb;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 4/27/16.
 */
public class PlugInF1Inspector {

    public static String simplePredictionAnalysis(PlugInF1 plugInF1, MultiLabelClfDataSet dataSet,
                                                  int dataPointIndex,  double classProbThreshold){
        StringBuilder sb = new StringBuilder();
        int numClasses = plugInF1.getNumClasses();
        MultiLabel trueLabels = dataSet.getMultiLabels()[dataPointIndex];
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        double[] classProbs = plugInF1.getImlGradientBoosting().predictClassProbs(dataSet.getRow(dataPointIndex));
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        List<Pair<Integer,Double>> list = IntStream.range(0,numClasses).mapToObj(l -> new Pair<Integer, Double>(l, classProbs[l]))
                .filter(pair -> pair.getSecond() >= classProbThreshold).sorted(comparator.reversed()).collect(Collectors.toList());
        for (Pair<Integer,Double> pair: list){
            int label = pair.getFirst();
            double prob = pair.getSecond();
            int match = 0;
            if (trueLabels.matchClass(label)){
                match=1;
            }
            sb.append(id).append("\t").append(labelTranslator.toExtLabel(label)).append("\t")
                    .append("single").append("\t").append(prob)
                    .append("\t").append(match).append("\n");
        }
        MultiLabel predicted = plugInF1.predict(dataSet.getRow(dataPointIndex));
        double probability = plugInF1.getImlGradientBoosting().predictAssignmentProb(dataSet.getRow(dataPointIndex),predicted);
        List<Integer> predictedList = predicted.getMatchedLabelsOrdered();
        sb.append(id).append("\t");
        for (int i=0;i<predictedList.size();i++){
            sb.append(labelTranslator.toExtLabel(predictedList.get(i)));
            if (i!=predictedList.size()-1){
                sb.append(",");
            }
        }
        sb.append("\t");
        int setMatch = 0;
        if (predicted.equals(trueLabels)){
            setMatch=1;
        }
        sb.append("set").append("\t").append(probability).append("\t").append(setMatch).append("\n");
        return sb.toString();
    }
}
