package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.Ngram;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Rainicy on 8/3/15.
 */
public class MekaFormat {


    public static void save(MultiLabelClfDataSet dataSet, String mekaFile, Config config) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(mekaFile));

        // writing the header: @relation 'data_name: -C number_classes\n\n'
        String dataName = config.getString("data.name");
        // starting writing labels
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        int numClasses = labelTranslator.getNumClasses();
        bw.write("@relation " + "'" + dataName + ": -C " + numClasses + "'\n\n");
        for (int i=0; i<numClasses; i++) {
            String labelName = labelTranslator.toExtLabel(i);
            bw.write("@attribute " + labelName.replace(" ", "_") + " {0,1}\n");
        }

        // starting writing features
        FeatureList featureList = dataSet.getFeatureList();
        Pattern pattern = Pattern.compile("ngram=(.*?), field");
        for (int i=0; i<featureList.size(); i++) {
            Feature feature = featureList.get(i);
            String featureName = "";
            if (feature instanceof Ngram) {
                Ngram ngram = (Ngram) feature;
                featureName = ngram.getNgram();
            }
            bw.write("@attribute " + featureName + " numeric\n");
        }

        // starting @data
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        bw.write("\n@data\n\n");
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            StringBuffer stringBuffer = new StringBuffer();
            stringBuffer.append("{");
            Vector rowData = dataSet.getRow(i);
            MultiLabel multiLabel = multiLabels[i];

            //starting with labels index.
            List<Integer> matchedLabels = multiLabel.getMatchedLabelsOrdered();
            for (int j=0; j<matchedLabels.size(); j++) {
                int matchedLabel = matchedLabels.get(j);
                if (j != 0) {
                    stringBuffer.append(",");
                }
                stringBuffer.append(matchedLabel + " " + "1");
            }
            // following by feature index
            Map<Integer, Double> sortedKeys = new TreeMap<>();
            for (Vector.Element element : rowData.nonZeroes()) {
                int index = element.index();
                double value = element.get();
                sortedKeys.put(index, value);
            }
            for (Map.Entry<Integer, Double> entry : sortedKeys.entrySet()) {
                int index = entry.getKey();
                double value = entry.getValue();
//                System.out.println("old index: " + index);
                index += numClasses;
//                System.out.println("new index: " + index);
//                System.in.read();
                stringBuffer.append("," + index + " " + value);
            }
            stringBuffer.append("}\n");
            bw.write(stringBuffer.toString());
        }
        bw.close();
    }
}
