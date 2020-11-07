import numpy as np
from numpy import save

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
try:
    from sklearn.datasets._twenty_newsgroups import (
        strip_newsgroup_footer, strip_newsgroup_quoting)
except ImportError:
    # scikit-learn < 0.24
    from sklearn.datasets.twenty_newsgroups import (
        strip_newsgroup_footer, strip_newsgroup_quoting)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib

# limit the list of categories to make running this example faster.
categories = ['rec.autos', 'rec.motorcycles']
train = fetch_20newsgroups(random_state=1,
                           subset='train',
                           categories=categories,
                           )
test = fetch_20newsgroups(random_state=1,
                          subset='test',
                          categories=categories,
                          )

class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.
    Takes a sequence of strings and produces a dict of sequences. Keys are
    `subject` and `body`.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # construct object dtype array with two columns
        # first column = 'subject' and second column = 'body'
        features = np.empty(shape=(len(posts), 2), dtype=object)
        for i, text in enumerate(posts):
            headers, _, bod = text.partition('\n\n')
            bod = strip_newsgroup_footer(bod)
            bod = strip_newsgroup_quoting(bod)
            features[i, 1] = bod

            prefix = 'Subject:'
            sub = ''
            for line in headers.split('\n'):
                if line.startswith(prefix):
                    sub = line[len(prefix):]
                    break
            features[i, 0] = sub

        return features


train_data = SubjectBodyExtractor().fit_transform(train.data)
test_data = SubjectBodyExtractor().fit_transform(test.data)

pipeline = Pipeline([
    ('union', ColumnTransformer(
        [
            ('subject', TfidfVectorizer(min_df=50), 0),

            ('body_bow', Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('best', TruncatedSVD(n_components=50)),
            ]), 1),

            # Removed from the original example as
            # it requires a custom converter.
            # ('body_stats', Pipeline([
            #   ('stats', TextStats()),  # returns a list of dicts
            #   ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            # ]), 1),
        ],

        transformer_weights={
            'subject': 0.8,
            'body_bow': 0.5,
            # 'body_stats': 1.0,
        }
    )),

    # Use a LogisticRegression classifier on the combined features.
    # Instead of LinearSVC (not fully ready in onnxruntime).
    ('logreg', LogisticRegression()),
])

pipeline.fit(train_data, train.target)
print(classification_report(pipeline.predict(test_data), test.target))

save('train_data.npy',train_data)
joblib.dump(pipeline,"output/model.pkl",compress=9)

