package edu.neu.ccs.pyramid.esplugins;

import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;

import java.io.IOException;

/** Expert: A <code>Scorer</code> for documents matching a <code>Term</code>.
 */
final class TermCountScorer extends Scorer {
    private final PostingsEnum postingsEnum;

    /**
     * Construct a <code>TermCountScorer</code>.
     *
     * @param weight
     *          The weight of the <code>Term</code> in the query.
     * @param td
     *          An iterator over the documents matching the <code>Term</code>.
     */
    TermCountScorer(Weight weight, PostingsEnum td) {
        super(weight);
        this.postingsEnum = td;
    }

    @Override
    public int docID() {
        return postingsEnum.docID();
    }

    @Override
    public int freq() throws IOException {
        return postingsEnum.freq();
    }

    @Override
    public DocIdSetIterator iterator() {
        return postingsEnum;
    }

    @Override
    public float score() throws IOException {
        assert docID() != DocIdSetIterator.NO_MORE_DOCS;
        return postingsEnum.freq();
    }

    /** Returns a string representation of this <code>TermCountScorer</code>. */
    @Override
    public String toString() { return "scorer(" + weight + ")[" + super.toString() + "]"; }
}
