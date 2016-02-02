package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * produce cluster vs performance txt file
 * Created by chengli on 2/1/16.
 */
public class Exp147 {
    public static void main(String[] args) throws Exception{
        // nuswide
//        String input = "/Users/chengli/Downloads/CRF vs MEKA - cluster_nuswide.tsv";
//        File output = new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/results/cluster/nuswide/result.txt");

        // scene

        String input = "/Users/chengli/Downloads/CRF vs MEKA - cluster_scene.tsv";
        File output = new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/results/cluster/scene/result.txt");

        List<String> lines = FileUtils.readLines(new File(input));
        List<Double> clusters = new ArrayList<>();
        List<Double> accs = new ArrayList<>();
        List<Double> overlaps = new ArrayList<>();
        List<Double> hammings = new ArrayList<>();
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (i==0){
                continue;
            }
            String[] split = line.split("\t");
            clusters.add(Double.parseDouble(split[0]));
            accs.add(Double.parseDouble(split[2]));
            overlaps.add(Double.parseDouble(split[3]));
            hammings.add(Double.parseDouble(split[4]));
        }
        FileUtils.writeStringToFile(output,clusters.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""));
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,accs.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,overlaps.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,hammings.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);

    }
}
