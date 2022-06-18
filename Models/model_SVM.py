# cumlのSVCのラッパークラス
# predictの出力ラベルの型がfloat64のため，int型の正解ラベルに対して正常な評価ができず，
# sklearnのGridsearchCVが動かなかった．
# cumlのSVCの代わりにこのクラスを使用すれば，GridsearchCVが使用可能．

from cuml.svm import SVC

class MySVC(SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def predict(self, X):
        y_pred = super().predict(X).astype(int)
        return y_pred

def build_model(**kwargs):

    return MySVC(**kwargs)
