package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInspector;
import edu.neu.ccs.pyramid.util.Serialization;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;

/**
 * visualize clusters
 * Created by chengli on 1/11/16.
 */
public class Exp220 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        BMMClassifier bmmClassifier = (BMMClassifier)Serialization.deserialize(config.getString("input.model"));
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.dataSet"),
                DataSetType.ML_CLF_SPARSE, true);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();

        List<Map<MultiLabel,Double>> list = BMMInspector.visualizeClusters(bmmClassifier,dataSet);
        for (int k=0;k<list.size();k++){
            System.out.println("--------------------------------------");
            System.out.println("for cluster "+k);
            Comparator<Map.Entry<MultiLabel,Double>> comparator = Comparator.comparing(Map.Entry::getValue);
            List<Map.Entry<MultiLabel,Double>> sorted = list.get(k).entrySet().stream().sorted(comparator.reversed())
                    .collect(Collectors.toList());
            for (Map.Entry<MultiLabel,Double> entry: sorted){
                System.out.print(entry.getKey().toStringWithExtLabels(labelTranslator));
                System.out.print(":");
                System.out.print(entry.getValue());
                System.out.print(", ");
            }
            System.out.println();

        }
    }
}
