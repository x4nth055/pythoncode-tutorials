from sklearn import metrics

y_true = [5, 3, 5, 4, 4, 5]
y_pred = [3, 5, 5, 4, 3, 4]
# homogeneity: each cluster contains only members of a single class.
print(metrics.homogeneity_score(y_true, y_pred))
# completeness: all members of a given class are assigned to the same cluster.
print(metrics.completeness_score(y_true, y_pred))
# v-measure: harmonic mean of homogeneity and completeness
print(metrics.v_measure_score(y_true, y_pred))
