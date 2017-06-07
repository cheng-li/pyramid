package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 8/20/16.
 */
public class CRFInspector {

    public static String pairRelations(CMLCRF crf){
        List<CRFInspector.PairWeight> list = crf.getWeights().printPairWeights();
        Comparator<PairWeight> comparator = Comparator.comparing(pairWeight -> Math.abs(pairWeight.weight));
        List<CRFInspector.PairWeight> sorted = list.stream().sorted(comparator.reversed()).collect(Collectors.toList());
        StringBuilder sb = new StringBuilder();
        LabelTranslator labelTranslator = crf.getLabelTranslator();
        for (CRFInspector.PairWeight pairWeight: sorted){
            sb.append(labelTranslator.toExtLabel(pairWeight.label1)).append(":").append(pairWeight.hasLabel1).append(", ").append(labelTranslator.toExtLabel(pairWeight.label2))
                    .append(":").append(pairWeight.hasLabel2).append("=>").append(pairWeight.weight).append("\n");
        }
        return sb.toString();
    }


    public static  String simplePredictionAnalysis(CMLCRF crf,
                                                   PluginPredictor<CMLCRF> pluginPredictor,
                                                   MultiLabelClfDataSet dataSet,
                                                   int dataPointIndex,  double classProbThreshold){
        StringBuilder sb = new StringBuilder();
        MultiLabel trueLabels = dataSet.getMultiLabels()[dataPointIndex];
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        double[] combProbs = crf.predictCombinationProbs(dataSet.getRow(dataPointIndex));
        double[] classProbs = crf.calClassProbs(combProbs);
        MultiLabel predicted = pluginPredictor.predict(dataSet.getRow(dataPointIndex));

        List<Integer> classes = new ArrayList<Integer>();
        for (int k = 0; k < crf.getNumClasses(); k++){
            if (classProbs[k]>=classProbThreshold
                    ||dataSet.getMultiLabels()[dataPointIndex].matchClass(k)
                    ||predicted.matchClass(k)){
                classes.add(k);
            }
        }

        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        List<Pair<Integer,Double>> list = classes.stream().map(l -> new Pair<Integer, Double>(l, classProbs[l]))
                .sorted(comparator.reversed()).collect(Collectors.toList());
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


        double probability = 0;
        List<MultiLabel> support = crf.getSupportCombinations();
        for (int i=0;i<support.size();i++){
            MultiLabel candidate = support.get(i);
            if (candidate.equals(predicted)){
                probability = combProbs[i];
                break;
            }
        }

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


    public static class PairWeight{
        int label1;
        int label2;
        boolean hasLabel1;
        boolean hasLabel2;
        double weight;

        public PairWeight(int label1, int label2, boolean hasLabel1, boolean hasLabel2, double weight) {
            this.label1 = label1;
            this.label2 = label2;
            this.hasLabel1 = hasLabel1;
            this.hasLabel2 = hasLabel2;
            this.weight = weight;
        }
    }
}
