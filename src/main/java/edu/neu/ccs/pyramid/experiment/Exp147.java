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

//        String[] input = {"/Users/chengli/Downloads/CRF vs MEKA - cluster_scene.tsv",
//                "/Users/chengli/Downloads/CRF vs MEKA - cluster_scene_run2.tsv",
//                "/Users/chengli/Downloads/CRF vs MEKA - cluster_scene_run3.tsv"
//        };
//        File output = new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/results/cluster/scene/result.txt");

        // rcv1
//        String[] input = {"/Users/chengli/Downloads/CRF vs MEKA - cluster_rcv1.tsv"};
//        File output = new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/results/cluster/rcv1/result.txt");

        // mediamill
        String[] input = {"/Users/chengli/Downloads/CRF vs MEKA - cluster_mediamill.tsv"};
        File output = new File("/Users/chengli/Documents/papers/LTR/ICML16_mixture/results/cluster/mediamill/result.txt");


        List<Double> clusters = parseCluster(input[0]);
        List<List<Double>> accLists = new ArrayList<>();
        for (String s: input){
            accLists.add(parseAcc(s));
        }
        List<Double> accs = aver(accLists);

        List<List<Double>> overLists = new ArrayList<>();
        for (String s: input){
            overLists.add(parseOver(s));
        }
        List<Double> overlaps = aver(overLists);

        List<List<Double>> hamLists = new ArrayList<>();
        for (String s: input){
            hamLists.add(parseHam(s));
        }
        List<Double> hammings = aver(hamLists);


        FileUtils.writeStringToFile(output,clusters.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""));
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,accs.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,overlaps.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);
        FileUtils.writeStringToFile(output,"\n",true);
        FileUtils.writeStringToFile(output,hammings.toString().replaceAll(Pattern.quote("["),"").replaceAll(Pattern.quote("]"),""),true);

    }

    private static List<Double> parseCluster(String input) throws Exception{
        List<Double> clusters = new ArrayList<>();
        List<String> lines = FileUtils.readLines(new File(input));
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (i==0){
                continue;
            }
            String[] split = line.split("\t");
            clusters.add(Double.parseDouble(split[0]));

        }
        return clusters;
    }

    private static List<Double> parseAcc(String input) throws Exception{
        List<String> lines = FileUtils.readLines(new File(input));
        List<Double> accs = new ArrayList<>();
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (i==0){
                continue;
            }
            String[] split = line.split("\t");
            accs.add(Double.parseDouble(split[2]));
        }
        System.out.println(accs);
        return accs;
    }


    private static List<Double> parseOver(String input) throws Exception{
        List<String> lines = FileUtils.readLines(new File(input));
        List<Double> overlaps = new ArrayList<>();
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (i==0){
                continue;
            }
            String[] split = line.split("\t");
            overlaps.add(Double.parseDouble(split[3]));
        }
        return overlaps;
    }

    private static List<Double> parseHam(String input) throws Exception{
        List<String> lines = FileUtils.readLines(new File(input));
        List<Double> hammings = new ArrayList<>();
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i);
            if (i==0){
                continue;
            }
            String[] split = line.split("\t");
            hammings.add(Double.parseDouble(split[4]));
        }
        return hammings;
    }


    private static List<Double> aver(List<List<Double>> lists){
        int size = lists.get(0).size();
        double[] av = new double[size];
        for (int i=0;i<lists.size();i++){
            for (int j=0;j<size;j++){
                av[j] += lists.get(i).get(j);
            }
        }

        for (int j=0;j<size;j++){
            av[j] =av[j]/lists.size();
        }


        List<Double> aver = new ArrayList<>();
        for (int i=0;i<size;i++){
            aver.add(av[i]);
        }
        System.out.println(aver);
        return aver;

    }
}
