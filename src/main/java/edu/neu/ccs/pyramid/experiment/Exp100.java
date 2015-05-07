package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.Ngram;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

/**
 * check dataset information
 * Created by chengli on 5/6/15.
 */
public class Exp100 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("please specify the config file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        show(config);
    }

    public static void show(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);


        System.out.println("density = "+ DataSetUtil.density(dataSet));

        List<Ngram> ngrams = dataSet.getFeatureList().getAll().stream()
                .filter(feature -> feature instanceof Ngram)
                .map(feature -> (Ngram) feature).collect(Collectors.toList());

        int maxN = ngrams.stream().mapToInt(Ngram::getN).max().getAsInt();
        int maxSlop = ngrams.stream().mapToInt(Ngram::getSlop).max().getAsInt();
        double[][] counts = new double[maxN][maxSlop+1];
        ngrams.stream().forEach(ngram ->{
            int n = ngram.getN();
            int slop = ngram.getSlop();
            counts[n-1][slop] += 1;
        });

        double total = ngrams.size();
        System.out.println("total = "+total);
        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                counts[i][j] /= total;
            }
        }

        for (int i=0;i<maxN;i++){
            for (int j=0;j<maxSlop+1;j++){
                System.out.println("n="+(i+1)+", slop="+j+", p="+counts[i][j]);
            }
        }
    }

}
