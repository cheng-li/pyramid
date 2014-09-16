package edu.neu.ccs.pyramid.feature_extraction;

import org.elasticsearch.action.search.SearchResponse;

/**
 * Created by chengli on 9/15/14.
 */
public class PhraseInfo {
    private String phrase;
    private SearchResponse searchResponse;

    public SearchResponse getSearchResponse() {
        return searchResponse;
    }

    public void setSearchResponse(SearchResponse searchResponse) {
        this.searchResponse = searchResponse;
    }

    public PhraseInfo(String phrase) {
        this.phrase = phrase;
    }

    public String getPhrase() {
        return phrase;
    }

    /**
     * just connect strings, need search again
     * @param left
     * @param right
     * @return
     */
    public static PhraseInfo connect(PhraseInfo left, PhraseInfo right){
        String[] leftSplit = left.phrase.split(" ");
        String[] rightSplit = right.phrase.split(" ");
        if (!leftSplit[leftSplit.length-1].equals(rightSplit[0])){
            throw new IllegalArgumentException("don't share the same term, cannot connect");
        }
        int pivotLength = rightSplit[0].length();
        String connected = "";
        connected = connected.concat(left.phrase);
        connected = connected.concat(right.phrase.substring(pivotLength));
        return new PhraseInfo(connected);
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PhraseInfo that = (PhraseInfo) o;

        if (!phrase.equals(that.phrase)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        return phrase.hashCode();
    }

    @Override
    public String toString() {
        return "PhraseInfo{" +
                "phrase='" + phrase + '\'' +
                '}';
    }
}
