package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.index.query.MatchQueryBuilder;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 9/13/14.
 */
public class DFStats implements Serializable {
    private static final long serialVersionUID = 1L;
    private int numClasses;
    private Map<String, DFStat> phraseStatsMap;
    private List<List<DFStat>> sortedForClasses;

    public DFStats(int numClasses) {
        this.numClasses = numClasses;
        //must support multi-threading
        this.phraseStatsMap = new ConcurrentHashMap<>();
        this.sortedForClasses = new ArrayList<>(numClasses);
        for (int k=0;k<numClasses;k++){
            this.sortedForClasses.add(new ArrayList<DFStat>());
        }

    }

    public void update(ESIndex esIndex) throws IOException {
        IntStream.range(0, esIndex.getNumDocs()).parallel()
                .forEach(id -> this.updateByOneDoc(esIndex, "" + id));
    }

    public void update(ESIndex esIndex, String[] docids) throws IOException{
        Arrays.stream(docids).parallel()
                .forEach(id -> this.updateByOneDoc(esIndex, "" + id));
    }

    /**
     * just handle exception, nothing else
     * @param esIndex
     * @param docid
     */
    public void updateByOneDoc(ESIndex esIndex, String docid){
        try {
            updateWithException(esIndex,docid);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * update dfStats using docid
     * @param esIndex
     * @param docid
     */
    private void updateWithException(ESIndex esIndex, String docid) throws IOException {
        TermVectorResponse response = esIndex.getClient()
                .prepareTermVector(esIndex.getIndexName(), "document", docid)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields("body").
                        execute().actionGet();

        Terms terms = response.getFields().terms("body");
        if (terms==null){
            throw new NullPointerException(docid+"has no term in the body field");
        }
        TermsEnum iterator = terms.iterator(null);
        for (int i=0;i<terms.size();i++) {
            String term = iterator.next().utf8ToString();
            if (!containsPhrase(term)) {
                DFStat dfStat = new DFStat(this.numClasses, term);
                long df = esIndex.DF("body", term, MatchQueryBuilder.Operator.AND);
                dfStat.setDf(df);
                //todo can query only once, and then loop
                for (int k = 0; k < numClasses; k++) {
                    long dfForClass = esIndex.DFForClass("body", term,
                            MatchQueryBuilder.Operator.AND, esIndex.getLabelField(), k);
                    dfStat.setDfForClass(dfForClass, k);
                }
                this.put(dfStat);
            }
        }
    }

    public boolean containsPhrase(String phrase){
        return this.phraseStatsMap.containsKey(phrase);
    }

    /**
     * it is possible to put the same dfStat several times
     * @param dfStat
     */
    public void put(DFStat dfStat){
        this.phraseStatsMap.put(dfStat.getPhrase(),dfStat);
    }

    public void sort(){
        for(int k=0;k<numClasses;k++){
            sortForClass(k);
        }
    }

    private void sortForClass(int k){
        Comparator<DFStat> comparator = Comparator
                .comparing(dfStat -> dfStat.getPurityForClass(k));
        List<DFStat> list = this.phraseStatsMap.entrySet().parallelStream().map(Map.Entry::getValue)
                .sorted(comparator.reversed()).collect(Collectors.toList());
        //just add
        this.sortedForClasses.set(k,list);
    }

    public List<DFStat> getSortedDFStats(int classIndex){
        return this.sortedForClasses.get(classIndex);
    }


    //todo make this fast for repeated access, should not filter each time
    public List<DFStat> getSortedDFStats(int classIndex, int dfThreshold){
        return this.sortedForClasses.get(classIndex).stream()
                .filter(dfStat -> dfStat.getDf()>dfThreshold)
                .collect(Collectors.toList());
    }

    public List<String> getSortedTerms(int classIndex, int dfThreshold, int top){
        return this.sortedForClasses.get(classIndex).stream()
                .filter(dfStat -> dfStat.getDf()>dfThreshold)
                .map(DFStat::getPhrase)
                .limit(top)
                .collect(Collectors.toList());
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("numClasses").append("=").append(numClasses).append("\n");
        for (Map.Entry<String, DFStat> entry : phraseStatsMap.entrySet()) {
            sb.append(entry.getValue()).append("\n");
        }
        return sb.toString();
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

    public static DFStats deserialize(File file) throws Exception{
        try(
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
        ){
            DFStats dfStats = (DFStats)objectInputStream.readObject();
            return dfStats;
        }
    }

    public void serialize(String file) throws Exception{
        serialize(new File(file));
    }

    public static DFStats deserialize(String file) throws Exception{
        return deserialize(new File(file));
    }


}
