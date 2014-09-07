package edu.neu.ccs.pyramid.feature;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/7/14.
 */
public class FeatureMappers implements Serializable {
    private static final long serialVersionUID = 1L;
    private List<CategoricalFeatureMapper> categoricalFeatureMappers;
    private List<NumericalFeatureMapper> numericalFeatureMappers;
    private Map<Integer, String> nameMap;
    /**
     * last feature index occupied
     */
    private int lastFeatureIndex;

    private int totalDim;

    public FeatureMappers() {
        this.categoricalFeatureMappers = new ArrayList<>();
        this.numericalFeatureMappers = new ArrayList<>();
        this.lastFeatureIndex = -1;
        this.totalDim = 0;
        this.nameMap = new HashMap<>();
    }

    public int getLastFeatureIndex() {
        return lastFeatureIndex;
    }

    public int getAvailableFeatureIndex(){
        return lastFeatureIndex+1;
    }

    /**
     *
     * @return total dimensions
     * one categorical feature may occupy several dimensions
     */
    public int getTotalDim() {
        return totalDim;
    }

    public void addMapper(CategoricalFeatureMapper categoricalFeatureMapper){
        this.categoricalFeatureMappers.add(categoricalFeatureMapper);
        this.totalDim += categoricalFeatureMapper.getNumCategories();
        if (categoricalFeatureMapper.getEnd()> this.lastFeatureIndex){
            this.lastFeatureIndex = categoricalFeatureMapper.getEnd();
        }
        String featureName = categoricalFeatureMapper.getFeatureName();
        Map<Integer, String> indexCategoryMap = categoricalFeatureMapper.getIndexCategoryMap();
        for (Map.Entry<Integer, String> entry: indexCategoryMap.entrySet()){
            Integer featureIndex = entry.getKey();
            String category = entry.getValue();
            String name = featureName+"["+category+"]";
            this.nameMap.put(featureIndex,name);
        }

    }

    public void addMapper(NumericalFeatureMapper numericalFeatureMapper){
        this.numericalFeatureMappers.add(numericalFeatureMapper);
        this.totalDim += 1;
        if (numericalFeatureMapper.getFeatureIndex()> this.lastFeatureIndex){
            this.lastFeatureIndex = numericalFeatureMapper.getFeatureIndex();
        }
        String featureName = numericalFeatureMapper.getFeatureName();
        int featureIndex = numericalFeatureMapper.getFeatureIndex();
        this.nameMap.put(featureIndex,featureName);
    }

    public List<CategoricalFeatureMapper> getCategoricalFeatureMappers() {
        return categoricalFeatureMappers;
    }

    public List<NumericalFeatureMapper> getNumericalFeatureMappers() {
        return numericalFeatureMappers;
    }


    public String getName(int featureIndex){
        return this.nameMap.get(featureIndex);
    }

    public List<String> getAll(){
        return IntStream.range(0, totalDim).mapToObj(this::getName)
                .collect(Collectors.toList());
    }

    public void serialize(File file) throws Exception{
        File parent = file.getParentFile();
        if (!parent.exists()){
            parent.mkdirs();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
        ){
            objectOutputStream.writeObject(this);
        }
    }

    public static FeatureMappers deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            FeatureMappers featureMappers = (FeatureMappers)objectInputStream.readObject();
            return featureMappers;
        }
    }


    public void serialize(String file) throws Exception{
        serialize(new File(file));
    }

    public static FeatureMappers deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }

    @Override
    public String toString() {
        return "FeatureMappers{" +
                "nameMap=" + nameMap +
                '}';
    }
}
