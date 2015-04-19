package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;


import java.io.File;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * dump features for all queries in trec 8
 * Created by chengli on 4/18/15.
 */
public class Exp84 {
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
                Exp83.mainFromConfig(perQidConfig);
            }

        }

    }

    static Config perQidConfig(Config config, int qid){
        Config perQidConfig = new Config();
        Config.copy(config,perQidConfig);
        perQidConfig.setInt("qid",qid);
        String archive = config.getString("archive.folder");
        String qidArchive = (new File(archive,""+qid)).getAbsolutePath();
        perQidConfig.setString("archive.folder",qidArchive);

        return perQidConfig;
    }
}
