import lang2vec.lang2vec as l2v

FEATURES = 'syntax_average+phonology_average+inventory_average'
LANGUAGES = 'sv zh ja it ko eu en fi tr hi ru ar he akk aii bho krl koi kpv olo mr mdf sa tl wbp yo gsw am bm be br bxr yue myv fo kk kmr gun pcm ta te hsb cy'

features_avg = l2v.get_features(LANGUAGES, FEATURES)

missing_features = open('missing_features.txt', 'w', encoding='utf-8')
for lang, feat in features_avg.items():
    nof = feat.count('--')
    missing_features.write('{} : {}\n'.format(lang, nof))

missing_features.close()