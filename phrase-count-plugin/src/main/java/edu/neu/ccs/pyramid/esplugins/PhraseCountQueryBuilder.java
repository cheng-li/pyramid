package edu.neu.ccs.pyramid.esplugins;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CachingTokenFilter;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.elasticsearch.common.ParseField;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentParser;
import org.elasticsearch.index.query.AbstractQueryBuilder;
import org.elasticsearch.index.query.QueryParseContext;
import org.elasticsearch.index.query.QueryShardContext;
import org.elasticsearch.index.search.MatchQuery;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public class PhraseCountQueryBuilder extends AbstractQueryBuilder<PhraseCountQueryBuilder> {
    private String analyzer;
    private int slop = 0;
    private final String fieldName;
    public static final ParseField SLOP_FIELD = new ParseField("slop", "phrase_slop");
    public static final ParseField ANALYZER_FIELD = new ParseField("analyzer");
    public static final ParseField QUERY_FIELD = new ParseField("query");
    public static final ParseField IN_ORDER_FIELD = new ParseField("in_order");
    public static final ParseField WEIGHTED_COUNT_FIELD = new ParseField("weighted_count");


    private final String value;
    private boolean weightedCount = false;
    private boolean inOrder = false;

    public static final String NAME = "phrase_count_query";

    public PhraseCountQueryBuilder(String fieldName, Object value) {
        if (Strings.isEmpty(fieldName)) {
            throw new IllegalArgumentException("[" + NAME + "] requires fieldName");
        }
        if (value == null) {
            throw new IllegalArgumentException("[" + NAME + "] requires query value");
        }
        this.fieldName = fieldName;
        this.value = value.toString();
    }

    public PhraseCountQueryBuilder(String fieldName, int slop, boolean inOrder, boolean weightedCount, String... terms) {
        if (Strings.isEmpty(fieldName)) {
            throw new IllegalArgumentException("[" + NAME + "] requires fieldName");
        }
        if (terms == null) {
            throw new IllegalArgumentException("[" + NAME + "] requires terms");
        }
        this.fieldName = fieldName;
        this.value = String.join(" ", terms);
        this.inOrder = inOrder;
        this.weightedCount = weightedCount;
        this.slop = slop;
    }

    public PhraseCountQueryBuilder(StreamInput in) throws IOException {
        super(in);
        fieldName = in.readString();
        value = in.readString();
        slop = in.readVInt();
        inOrder = in.readBoolean();
        weightedCount = in.readBoolean();
        analyzer = in.readOptionalString();
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(fieldName);
        out.writeString(value);
        out.writeVInt(slop);
        out.writeBoolean(inOrder);
        out.writeBoolean(weightedCount);
        out.writeOptionalString(analyzer);
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.startObject(fieldName);

        builder.field(QUERY_FIELD.getPreferredName(), value);
        if (analyzer != null) {
            builder.field(ANALYZER_FIELD.getPreferredName(), analyzer);
        }
        builder.field(SLOP_FIELD.getPreferredName(), slop);
        builder.field(IN_ORDER_FIELD.getPreferredName(), inOrder);
        builder.field(WEIGHTED_COUNT_FIELD.getPreferredName(), weightedCount);
        printBoostAndQueryName(builder);
        builder.endObject();
        builder.endObject();
    }

    @Override
    protected boolean doEquals(PhraseCountQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName) &&
            Objects.equals(value, other.value) &&
            Objects.equals(analyzer, other.analyzer) &&
            Objects.equals(inOrder, other.inOrder) &&
            Objects.equals(weightedCount, other.weightedCount)
            && Objects.equals(slop, other.slop);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(fieldName, value, analyzer, slop, inOrder, weightedCount);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    /**
     * Explicitly set the analyzer to use. Defaults to use explicit mapping
     * config for the field, or, if not set, the default search analyzer.
     */
    public PhraseCountQueryBuilder analyzer(String analyzer) {
        this.analyzer = analyzer;
        return this;
    }

    /** Get the analyzer to use, if previously set, otherwise <tt>null</tt> */
    public String analyzer() {
        return this.analyzer;
    }

    /** Sets a slop factor for phrase queries */
    public PhraseCountQueryBuilder slop(int slop) {
        if (slop < 0) {
            throw new IllegalArgumentException("No negative slop allowed.");
        }
        this.slop = slop;
        return this;
    }

    /** Get the slop factor for phrase queries. */
    public int slop() {
        return this.slop;
    }

    public boolean weightedCount() {
        return this.weightedCount;
    }

    public void weightedCount(boolean weightedCount) {
        this.weightedCount = weightedCount;
    }

    public boolean inOrder() {
        return inOrder;
    }

    public void inOrder(boolean inOrder) {
        this.inOrder = inOrder;
    }

    protected Query doToQuery(QueryShardContext context) throws IOException {
//        Analyzer analyzer = context.getMapperService().searchAnalyzer();
        Analyzer analyzer = new WhitespaceAnalyzer();
        try (TokenStream source = analyzer.tokenStream(fieldName, value.toString())) {
            CachingTokenFilter stream = new CachingTokenFilter(new LowerCaseFilter(source));
            TermToBytesRefAttribute termAtt = stream.getAttribute(TermToBytesRefAttribute.class);
            if (termAtt == null) {
                return null;
            }
            List<CustomSpanTermQuery> clauses = new ArrayList<>();
            stream.reset();
            while (stream.incrementToken()) {
                Term term = new Term(fieldName, termAtt.getBytesRef());
                    clauses.add(new CustomSpanTermQuery(term));
            }
            return new PhraseCountQuery(clauses.toArray(new CustomSpanTermQuery[clauses.size()]), slop, inOrder, weightedCount);
        } catch (IOException e) {
            throw new RuntimeException("Error analyzing query text", e);
        }


    }
    private boolean isPosTag(String s) {
        return s.charAt(0) == '<' && s.charAt(s.length() - 1) == '>';
    }

    private String getPosTag(String s) {
        return s.substring(1, s.length() - 1);
    }
    //XSON (maps to Content-Type application/xson) is an optimized binary representation of JSON.
    public static Optional<PhraseCountQueryBuilder> fromXContent(QueryParseContext parseContext) throws IOException {
        XContentParser parser = parseContext.parser();
        String fieldName = null;
        Object value = null;
        float boost = AbstractQueryBuilder.DEFAULT_BOOST;
        String analyzer = null;
        int slop = MatchQuery.DEFAULT_PHRASE_SLOP;
        boolean inOrder = false;
        boolean weightedCount = false;
        String queryName = null;
        String currentFieldName = null;
        XContentParser.Token token;
        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                currentFieldName = parser.currentName();
            } else if (parseContext.isDeprecatedSetting(currentFieldName)) {
                // skip
            } else if (token == XContentParser.Token.START_OBJECT) {
                throwParsingExceptionOnMultipleFields(NAME, parser.getTokenLocation(), fieldName, currentFieldName);
                fieldName = currentFieldName;
                while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
                    if (token == XContentParser.Token.FIELD_NAME) {
                        currentFieldName = parser.currentName();
                    } else if (token.isValue()) {
                        if (PhraseCountQueryBuilder.QUERY_FIELD.match(currentFieldName)) {
                            value = parser.objectText();
                        } else if (ANALYZER_FIELD.match(currentFieldName)) {
                            analyzer = parser.text();
                        } else if(IN_ORDER_FIELD.match(currentFieldName)) {
                            inOrder = parser.booleanValue();
                        } else if (WEIGHTED_COUNT_FIELD.match(currentFieldName)) {
                            weightedCount = parser.booleanValue();
                        } else if (BOOST_FIELD.match(currentFieldName)) {
                            boost = parser.floatValue();
                        } else if (SLOP_FIELD.match(currentFieldName)) {
                            slop = parser.intValue();
                        } else if (AbstractQueryBuilder.NAME_FIELD.match(currentFieldName)) {
                            queryName = parser.text();
                        } else {
                            throw new ParsingException(parser.getTokenLocation(),
                                "[" + NAME + "] query does not support [" + currentFieldName + "]");
                        }
                    } else {
                        throw new ParsingException(parser.getTokenLocation(),
                            "[" + NAME + "] unknown token [" + token + "] after [" + currentFieldName + "]");
                    }
                }
            } else {
                throwParsingExceptionOnMultipleFields(NAME, parser.getTokenLocation(), fieldName, parser.currentName());
                fieldName = parser.currentName();
                value = parser.objectText();
            }
        }

        PhraseCountQueryBuilder phraseCountQuery = new PhraseCountQueryBuilder(fieldName, value);
        phraseCountQuery.analyzer(analyzer);
        phraseCountQuery.slop(slop);
        phraseCountQuery.inOrder(inOrder);
        phraseCountQuery.weightedCount(weightedCount);
        phraseCountQuery.queryName(queryName);
        phraseCountQuery.boost(boost);
        return Optional.of(phraseCountQuery);
    }

}

