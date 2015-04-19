package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * run logistic regression on all trec8 queries
 * Created by chengli on 4/18/15.
 */
public class Exp85 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        int[] goodQueries = {301, 304, 306, 307, 311, 313, 318, 319, 321, 324, 331, 332, 343, 346, 347, 352, 353, 354, 357, 360, 367, 370, 374, 376, 383, 389, 390, 391, 392, 395, 398, 399, 400, 401, 404, 408, 412, 415, 418, 422, 424, 425, 426, 428, 431, 434, 435, 436, 438, 439, 443, 446, 450};
        Set<Integer> goodQuerySet = Arrays.stream(goodQueries).mapToObj(i -> i).
                collect(Collectors.toSet());



        for (int qid=401;qid<=450;qid++){
            if (goodQuerySet.contains(qid)){
                System.out.println("=============================");
                System.out.println("qid = "+qid);
                Config perQidConfig = perQidConfig(config,qid);
                System.out.println(perQidConfig);
                Exp70.train(perQidConfig);
            }
        }

        List<Double> accuracies = new ArrayList<>();
        for (int qid=401;qid<=450;qid++){
            if (goodQuerySet.contains(qid)){
                System.out.println("=============================");
                System.out.println("qid = "+qid);
                Config perQidConfig = perQidConfig(config,qid);
                double accuracy = Exp70.test(perQidConfig);
                accuracies.add(accuracy);
            }

        }

        System.out.println("all done");
        System.out.println("average accuracy = "+accuracies.stream()
                .mapToDouble(acc -> acc)
        .average().getAsDouble());


    }

    static Config perQidConfig(Config config, int qid){
        Config perQidConfig = new Config();
        Config.copy(config,perQidConfig);
        perQidConfig.setInt("qid",qid);
        String input = config.getString("input.folder");
        String perQidInput = (new File(input,""+qid)).getAbsolutePath();
        perQidConfig.setString("input.folder",perQidInput);

        String archive = config.getString("output.folder");
        String qidArchive = (new File(archive,""+qid)).getAbsolutePath();
        perQidConfig.setString("output.folder",qidArchive);

        return perQidConfig;
    }
}
