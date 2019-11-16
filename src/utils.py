import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def plt_roc(label, prediction):
	tp, fp, _ = roc_curve(label, prediction)
	auc = roc_auc_score(label, prediction)
	plt.plot(tp, fp, label=f'Auc is {auc}')
	plt.show()
