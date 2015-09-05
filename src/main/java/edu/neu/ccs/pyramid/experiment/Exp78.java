package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature_selection.FusedKolmogorovFilter;
import edu.neu.ccs.pyramid.util.EmpiricalCDF;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * visualize Kolmogorov–Smirnov test
 * Created by chengli on 3/27/15.
 */
public class Exp78 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        selection(config);
    }

    private static void selection(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        int featureIndex = config.getInt("featureIndex");
        FusedKolmogorovFilter filter = new FusedKolmogorovFilter();
        Vector vector = dataSet.getColumn(featureIndex);
        List<List<Double>> inputs = filter.generateInputsEachClass(vector, dataSet.getLabels(), dataSet.getNumClasses());
        List<EmpiricalCDF> cdfs = filter.generateCDFs(vector,inputs);
        for (int k=0;k<dataSet.getNumClasses();k++){
            System.out.println("for class "+dataSet.getLabelTranslator().toExtLabel(k));
            System.out.println(cdfs.get(k));
        }
    }
}
