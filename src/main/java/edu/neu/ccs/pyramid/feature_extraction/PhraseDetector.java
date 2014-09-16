package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import org.elasticsearch.action.search.SearchResponse;

import java.util.*;

/**
 * Created by chengli on 9/13/14.
 */
public class PhraseDetector {
    private int minDf=10;
    private ESIndex index;

    public PhraseDetector(ESIndex index) {
        this.index = index;
    }

    public PhraseDetector setMinDf(int minDf) {
        this.minDf = minDf;
        return this;
    }

    public Set<PhraseInfo> getPhraseInfos(Map<Integer, String> termVector, Set<String> seeds){
        Set<PhraseInfo> phrases = new HashSet<>();
        for (Map.Entry<Integer,String> entry: termVector.entrySet()){
            int pos = entry.getKey();
            String term = entry.getValue();
            if (seeds.contains(term)){
                phrases.addAll(this.getPhraseInfos(termVector,pos));
            }
        }
        return phrases;
    }

    /**
     *
     * @param termVector
     * @param pos
     * @return
     */
    public Set<PhraseInfo> getPhraseInfos(Map<Integer, String> termVector, int pos){
        Set<PhraseInfo> phrases = new HashSet<>();
        List<PhraseInfo> leftPhrases = exploreLeft(termVector,pos);
        List<PhraseInfo> rightPhrases = exploreRight(termVector, pos);
        List<PhraseInfo> connected = connect(leftPhrases,rightPhrases);
        phrases.addAll(leftPhrases);
        phrases.addAll(rightPhrases);
        phrases.addAll(connected);
        return phrases;
    }

    /**
     * phrases ending with current term
     * @param termVector
     * @param pos
     * @return
     */
    public List<PhraseInfo> exploreLeft(Map<Integer, String> termVector, int pos){
        List<PhraseInfo> phraseInfos = new ArrayList<>();
        String center = termVector.get(pos);
        String phrase = center;
        int currentLeft = pos - 1;
        while(true){
            if(!termVector.containsKey(currentLeft)){
                break;
            }
            String leftTerm = termVector.get(currentLeft);
            String currentPhrase = leftTerm.concat(" ").concat(phrase);
            SearchResponse searchResponse = index.matchPhrase(index.getBodyField(),
                    currentPhrase,0);
            if (searchResponse.getHits().totalHits()<this.minDf){
                break;
            }
            PhraseInfo phraseInfo = new PhraseInfo(currentPhrase);
            phraseInfo.setSearchResponse(searchResponse);
            phraseInfos.add(phraseInfo);
            phrase = currentPhrase;
            currentLeft -= 1;
        }
        return phraseInfos;
    }

    /**
     * phrases starting with current term
     * @param termVector
     * @param pos
     * @return
     */
    public List<PhraseInfo> exploreRight(Map<Integer, String> termVector, int pos){
        List<PhraseInfo> phraseInfos = new ArrayList<>();
        String center = termVector.get(pos);
        String phrase = center;
        int currentRight = pos + 1;
        while(true){
            if(!termVector.containsKey(currentRight)){
                break;
            }
            String rightTerm = termVector.get(currentRight);
            String currentPhrase = phrase.concat(" ").concat(rightTerm);
            SearchResponse searchResponse = index.matchPhrase(index.getBodyField(),
                    currentPhrase,0);
            if (searchResponse.getHits().totalHits()<this.minDf){
                break;
            }
            PhraseInfo phraseInfo = new PhraseInfo(currentPhrase);
            phraseInfo.setSearchResponse(searchResponse);
            phraseInfos.add(phraseInfo);
            phrase = currentPhrase;
            currentRight += 1;
        }
        return phraseInfos;
    }

    public List<PhraseInfo> connect(List<PhraseInfo> leftList, List<PhraseInfo> rightList){
        List<PhraseInfo> allConnected = new ArrayList<>();
        int numLeft = leftList.size();
        int numRight = rightList.size();
        boolean[][] valid = new boolean[numLeft][numRight];
        for (int i=0;i<numLeft;i++){
            PhraseInfo left = leftList.get(i);
            for (int j=0;j<numRight;j++){

                //just try it
                if (i==0){
                    PhraseInfo right = rightList.get(j);
                    PhraseInfo connected = PhraseInfo.connect(left,right);
                    SearchResponse searchResponse = index.matchPhrase(index.getBodyField(),
                            connected.getPhrase(),0);
                    if (searchResponse.getHits().totalHits()<this.minDf){
                        //skip j
                        break;
                    }
                    connected.setSearchResponse(searchResponse);
                    allConnected.add(connected);
                    valid[i][j] = true;
                } else {
                    //look at the row above it
                    if (!valid[i-1][j]){
                        break;
                    }
                    PhraseInfo right = rightList.get(j);
                    PhraseInfo connected = PhraseInfo.connect(left,right);
                    SearchResponse searchResponse = index.matchPhrase(index.getBodyField(),
                            connected.getPhrase(),0);
                    if (searchResponse.getHits().totalHits()<this.minDf){
                        //skip j
                        break;
                    }
                    connected.setSearchResponse(searchResponse);
                    allConnected.add(connected);
                    valid[i][j] = true;
                }
            }
        }
        return allConnected;
    }


}
