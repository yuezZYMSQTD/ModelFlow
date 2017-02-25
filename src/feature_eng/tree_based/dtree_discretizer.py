# -*- encoding:UTF-8 -*-
"""
文档基于sklearn 0.17.1
"""
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

__author__ = u'何君柯 <junkeh@princetechs.com>'
__version__ = '0.13.1'

class DTNode(object):
    """协助DTDiscretizer将sklearn的决策树中的tree_.feature转化成二叉树

    参数
    ----
    content: tuple
        该参数传入一个tuple: tuple的第一项为特征在原数据集中的列index,
        tuple的第二项为对应的决策树结点的阀值

    属性
    ----
    left_child_: DTNode
        该结点的左子结点
    right_child_: DTNode
        该结点的右子结点
    """
    def __init__(self, content):
        self.content = content
        self.left_child_ = None
        self.right_child_ = None

    def get_content(self):
        """获得结点的数据

        :return: tuple
            tuple的内容见class的参数部分针对self.content的说明
        """
        return self.content

    def get_left(self):
        """返回结点的左子结点

        :return: DTNode 结点的左子结点
        """
        return self.left_child_

    def get_right(self):
        """返回结点的右子结点

        :return: DTNode 结点的右子结点
        """
        return self.right_child_

    def set_left(self, left_child):
        """Modification函数,设置结点的左子结点

        :param left_child: DTNode
            该结点的左子结点
        """
        self.left_child_ = left_child

    def set_right(self, right_child):
        """Modification函数,设置结点的右子结点

        :param right_child: DTNode
            该结点的右子结点
        """
        self.right_child_ = right_child


class BaseDTDiscretizer(six.with_metaclass(ABCMeta,
                                           BaseEstimator,
                                           TransformerMixin)):
    paths_ = []
    to_str_indent_ = ' ' * 12
    n_col_per_line_ = 4 # 大于0

    @abstractmethod
    def __init__(self,
                 columns,
                 n_decimals,
                 v_convert_functions,
                 sep,
                 inters_only,
                 classification,
                 tree=None,
                 tree_=None,
                 max_depth=5,
                 **kwargs):
        """不可直接使用,应使用ClassificationDTDiscretizer或RegressionDTDiscretizer
        """
        self.columns = list(columns)
        self.v_convert_functions = v_convert_functions
        self.n_decimals = n_decimals
        self.sep = sep
        self.inters_only = inters_only
        self.classification = classification
        self.tree = tree
        self.tree_ = tree_
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.tree_inter_feature_names_ = None

    def reset_discretizer(self):
        """重置该DTDiscretizer

         Modification函数,无返回值
        """
        self.paths_ = []
        self.tree_inter_feature_names_ = None

    def fit(self, X, y):
        """DTDiscretizer对数据集X进行拟合

        :param X: array-like, shape = [n_samples, n_features]
            拟合self.tree使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :param y: array-like, shape = [n_samples]
            目标矩阵
        :return: self
        """
        X = np.array(X)
        y = np.array(y)

        n_sample = X.shape[0]
        decimal_form = '%.' + str(self.n_decimals) + 'f'

        # self.tree对X, y进行拟合
        if not self._dt_tree_fit(X, y):
            self.reset_discretizer()
            return None
        # 得到决策树交叉项的生成路径
        if not self._generate_paths():
            self.reset_discretizer()
            return None

        # 开始生成新特征名称
        self.tree_inter_feature_names_ = []

        i = 0
        for new_fea_path in self.paths_:
            name_tmp = {}
            for fea_thre, comp in zip(new_fea_path[0], new_fea_path[1]):
                fea_tmp = self.columns[fea_thre[0]]
                if (self.v_convert_functions is None) or (self.v_convert_functions.get(fea_tmp) is None):
                    thre_tmp = fea_thre[1]
                else:
                    thre_tmp = self.v_convert_functions[fea_tmp](fea_thre[1])

                if not fea_tmp in name_tmp.keys():
                    name_tmp[fea_tmp] = {comp: thre_tmp}
                elif comp in name_tmp[fea_tmp].keys():
                    name_tmp[fea_tmp][comp] = min(name_tmp[fea_tmp][comp], thre_tmp) if comp == '<=' else max(name_tmp[fea_tmp][comp], thre_tmp)
                else:
                    name_tmp[fea_tmp][comp] = thre_tmp

            fea_included_tmp = list(name_tmp.keys())
            fea_included_tmp.sort()

            nodes_combine = []
            for fea_name_tmp in fea_included_tmp:
                if len(name_tmp[fea_name_tmp].keys()) == 1:
                    k_v = list(name_tmp[fea_name_tmp].items())[0]
                    nodes_combine.append(fea_name_tmp + k_v[0] + (decimal_form % k_v[1]))
                else:
                    min_lim_tmp = (decimal_form % name_tmp[fea_name_tmp]['>'])
                    max_lim_tmp = (decimal_form % name_tmp[fea_name_tmp]['<='])
                    nodes_combine.append(min_lim_tmp + '<' + fea_name_tmp + '<=' + max_lim_tmp)
            self.tree_inter_feature_names_.append(self.sep.join(nodes_combine))
            i += 1
        return self

    def transform(self, X):
        """DTDiscretizer对X进行转化

        :param X: array-like [n_samples, n_features]
            拟合self.tree使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :return: np.array shape = [n_samples, (n_features+len(self.tree_inter_feature_names_))]
            对源数据进行转化后的数据集
        """
        X = np.array(X)
        n_samples = X.shape[0]

        if self.tree_inter_feature_names_ is None:
            print(u'请先对数据集使用fit函数,或直接调用fit_transform函数')
            return None

        inter_features = np.zeros((n_samples, len(self.tree_inter_feature_names_)))

        i = 0
        for new_fea_path in self.paths_:
            new_fea_tmp = np.ones(n_samples, dtype=bool)
            for fea_thre, comp in zip(new_fea_path[0], new_fea_path[1]):
                new_fea_tmp &= self._compare(X[:, fea_thre[0]], comp, fea_thre[1])
            inter_features[new_fea_tmp, i] = 1
            i += 1

        if self.inters_only:
            return inter_features
        else:
            return np.concatenate((X, inter_features), axis=1)

    def fit_transform(self, X, y):
        """DTDiscretizer先fit数据集X再对X进行transform,跟先调用fit再调用transform的结果无差别。
        然而不建议对测试集使用该函数。

        :param X: array-like [n_samples, n_features]
            拟合self.tree使用的数据集,建议对X中的名义变量使用独热编码后再传参
         :param y: array-like, shape = [n_samples]
            目标矩阵
        :return: np.array shape = [n_samples, (n_features+len(self.tree_inter_feature_names_))]
            对源数据进行转化后的数据集
        """
        self.fit(X, y)
        return self.transform(X)

    def get_column_names(self):
        """返回新数据集各特征名称

        :return: list
            新数据集各特征名称
        """
        if self.inters_only:
            return self.tree_inter_feature_names_
        else:
            return self.columns + self.tree_inter_feature_names_

    def _dt_tree_fit(self, X, y):
        """对self.tree使用X和y进行拟合

        :param X: array-like [n_samples, n_features]
            拟合self.tree使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :param y: array-like, shape = [n_samples]
            目标矩阵
        :return: True
        """
        if self.tree_ is not None:
            self.n_nodes_ = len(self.tree_[0])
            self.features_ = self.tree_[0]
            self.thresholds_ = self.tree_[1]
            self.v_leaf_ = self.tree_[0][-1]
            return True

        # 如果传入训练过的决策树,则使用该决策树,否则根据传入参数自建新的决策树
        if self.tree is not None:
            curr_tree = self.tree
        else:
            if self.classification:
                curr_tree = DecisionTreeClassifier(max_depth=self.max_depth, **self.kwargs)
            else:
                curr_tree = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            curr_tree.fit(X, y)

        self.n_nodes_ = curr_tree.tree_.node_count
        self.features_ = curr_tree.tree_.feature
        self.thresholds_ = curr_tree.tree_.threshold
        self.v_leaf_ = self.features_[-1]
        return True

    def _generate_paths(self):
        """生成所有决策树交叉项的生成路径

        :return: True,如果路径生成成功;False,如果生成失败
        """
        self.reset_discretizer()
        dt_root = self._construct_tree(self.features_, self.thresholds_)
        if dt_root is None:
            return False
        return self._dt_pre_order(dt_root, [], [])

    def _generate_node(self, fea_idx, threshold):
        """生成二叉树的结点

        :param fea_idx:
            决策树结点的特征在源数据中对应的列号
        :param threshold: float
            决策树结点的阀值
        :return: DTNode 或 None
            如果fea_idx!=self.v_leaf_,返回DTNode
            否则返回None
        """
        if fea_idx != self.v_leaf_:
            return DTNode((fea_idx, threshold))
        else:
            return None

    def _construct_tree(self, tree_features, tree_threshold):
        """将self.tree.tree_.feature和self.tree.tree_.threshold中的信息转化为二叉树的数据结构

        :param tree_features: np.array
            开发时参照 sklearn 0.17.1 的sklearn.tree的属性信息
        :param tree_threshold: np.array
            开发时参照 sklearn 0.17.1 的sklearn.tree的属性信息
        :return: DTNode
            二叉树的根结点
        """
        nodes_stack = list()

        if (self.n_nodes_ <= 0) | (tree_features[0] == self.v_leaf_):
            print(u"当前决策树节点数为0,无需根据决策树生成交叉项...")
            return None

        root = self._generate_node(tree_features[0], tree_threshold[0])

        # 对tree_.feature中的每一项进行遍历,当该项!=self.v_leaf_时,后一项是前一项的
        # 左子结点; 当该项==self.v_leaf_时,后一项是前面某项的右子节点,使用self._find_parent
        # 函数找出父结点
        curr_node = root
        for i in range(self.n_nodes_-1):
            child_tmp = self._generate_node(tree_features[i+1], tree_threshold[i+1])
            if curr_node is not None:
                curr_node.set_left(child_tmp)
                nodes_stack.append(curr_node)
            else:
                nodes_stack.pop().set_right(child_tmp)
            curr_node = child_tmp
        return root

    def _dt_pre_order(self, root, feature_path, compare_path):
        """对将决策树转化为的二叉树进行前序遍历,并记录所有属于其父结点的左子节点的
        路径信息

        :param root: DTNode
            二叉树的根结点
        :param feature_path: tuple组成的list
            到达root需经过的所有结点的content所形成的list,content的内容参见DTNode的
            self.content
        :param compare_path: list
            到达root需经过的所有决策树结点的对比方法(小于等于/大于)所组成的list
        :return: True 唯一返回值
            该函数为modification函数,会修改DTDiscretizer的self.paths_
        """
        if root is None:
            return True

        path_to_child = feature_path + [root.get_content()]
        to_left_compare = compare_path + ['<=']
        to_right_compare = compare_path + ['>']
        self.paths_.append((path_to_child, to_left_compare))

        self._dt_pre_order(root.get_left(), path_to_child, to_left_compare)
        self._dt_pre_order(root.get_right(), path_to_child, to_right_compare)
        return True

    @staticmethod
    def _compare(xcol, comp, value):
        """比较xcol中的元素和value的大小

        :param xcol: np.array [n_samples,]
            原数据中的列
        :param comp: str
            "<=" 或者 ">"
        :param value: float
            结点阀值
        :return: np.array [n_samples,]
            返回的np.array的dtype = bool
        """
        if comp == "<=":
            return xcol <= value
        else:
            return xcol > value

    def __str__(self):
        if self.classification:
            class_name = 'ClassificationDTDiscretizer'
        else:
            class_name = 'RegressionDTDiscretizer'

        return (class_name + '(n_decimals=%d, sep=\"%s\", inters_only=%s, max_depth=%d, \n' \
               + self.__format_kwargs__() \
               + self.to_str_indent_ + 'columns=%s)') \
               % (self.n_decimals, self.sep, self.inters_only, self.max_depth, self.__format_columns__())

    def __format_kwargs__(self):
        s = ''
        kwargs = self.kwargs

        i = 0
        for arg in kwargs:
            str_arg = kwargs[arg]
            str_arg = '\"' + str_arg + '\"' if isinstance(str_arg, str) else str(str_arg)

            if i == 0:
                s += (self.to_str_indent_ + arg + '=' + str_arg)
                i += 1
            elif i <= 2:
                s += (', ' + arg + '=' + str_arg)
                i += 1
            else:
                s += (', ' + arg + '=' + str_arg + ', \n')
                i = 0

        if len(s) > 0 and s[-3:] != ', \n':
            s += ', \n'
        return s

    def __format_columns__(self):
        list_indent = self.to_str_indent_ + ' ' * len('columns=[')
        s = ''

        i = 0
        for col in self.columns:
            if i == 0:
                s += list_indent + '\'' + col + '\''
                i += 1
            elif i < (self.n_col_per_line_-1):
                s += ', \'' + col + '\''
                i += 1
            else:
                s += ', \'' + col + '\', \n'
                i = 0
        s = s.strip()
        if s.endswith(', \n'):
            s = s[:s.rfind(', \n')]
        return '[' + s + ']'

    def __repr__(self):
        return self.__str__()


class ClassificationDTDiscretizer(BaseDTDiscretizer):
    """基于决策树的独热编码型交叉项生成器,可返回新特征及特征名称

        参数
        ----
        columns: array-like shape = [n_features]
            源数据集特征名称
        v_convert_functions: dict [特征名称: 转化函数]
            生成结点名称时,如果特征数据在传入前经过函数转化可在此处提供反函数,使得新生成特征
            名称的阀值为原数值
        n_decimals: int (default=3)
            生成的交叉项特征名称中threshold小数点后保留位数
        sep: str (default=',')
            在对新生成的交叉项进行命名时,discretizer对各结点条件进行组合使用的分隔符号
        inters_only: bool (default=False)
            是否只使用新生成的离散型交叉项
        tree: sklearn.tree
            训练好的决策树,该参数优先于除了tree_以外的决策树相关参数
        tree_: tuple (sklearn.tree_.feature, sklearn.tree_.threshold)
            sklearn决策树的feature和threshold属性。该参数的主要意义在于可以对训练完的决策树
            手动调整结点,但需要保证tuple中的两项保留了原有的格式。该参数为决策树相关参数的最优先
            使用参数
        max_depth: int
            决策树的最大深度。如tree或tree_参数已提供,该参数无效
        kwargs:
            还可提供决策树的其它参数,详情参见sklearn。如tree或tree_参数已提供,该参数无效

        属性
        ----
        tree_inter_feature_names_: list
            根据决策树生成的交叉项的特征名称的list
        n_nodes_: int
            决策树的节点数(包含None节点)
        features_: list
            sklearn.tree.tree_.feature
        thresholds_: list
            sklearn.tree.tree_.threshold
        v_leaf_: int
            sklearn.tree_.feature中表示决策树叶子的值,根据 sklearn 0.17.1 将
            sklearn.tree_.feature中的最后一项的值认为是决策树叶子的值
        paths_: list
            所有决策树交叉项的生成路径

        参考
        ----
        sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor
        """
    def __init__(self,
                 columns,
                 n_decimals=3,
                 v_convert_functions=None,
                 sep=', ',
                 inters_only=False,
                 tree=None,
                 tree_=None,
                 max_depth=5,
                 **kwargs):
        super(ClassificationDTDiscretizer, self).__init__(
            columns=columns,
            v_convert_functions=v_convert_functions,
            n_decimals=n_decimals,
            sep=sep,
            inters_only=inters_only,
            classification=True,
            tree=tree,
            tree_=tree_,
            max_depth=max_depth,
            **kwargs)


class RegressionDTDiscretizer(BaseDTDiscretizer):
    """基于决策树的独热编码型交叉项生成器,可返回新特征及特征名称

        参数
        ----
        columns: array-like shape = [n_features]
            源数据集特征名称
        v_convert_functions: dict [特征名称: 转化函数]
            生成结点名称时,如果特征数据在传入前经过函数转化可在此处提供反函数,使得新生成特征
            名称的阀值为原数值
        n_decimals: int (default=3)
            生成的交叉项特征名称中threshold小数点后保留位数
        sep: str (default=',')
            在对新生成的交叉项进行命名时,discretizer对各结点条件进行组合使用的分隔符号
        inters_only: bool (default=False)
            是否只使用新生成的离散型交叉项
        tree: sklearn.tree
            训练好的决策树,该参数优先于除了tree_以外的决策树相关参数
        tree_: tuple (sklearn.tree_.feature, sklearn.tree_.threshold)
            sklearn决策树的feature和threshold属性。该参数的主要意义在于可以对训练完的决策树
            手动调整结点,但需要保证tuple中的两项保留了原有的格式。该参数为决策树相关参数的最优先
            使用参数
        max_depth: int
            决策树的最大深度。如tree或tree_参数已提供,该参数无效
        kwargs:
            还可提供决策树的其它参数,详情参见sklearn。如tree或tree_参数已提供,该参数无效

        属性
        ----
        tree_inter_feature_names_: list
            根据决策树生成的交叉项的特征名称的list
        n_nodes_: int
            决策树的节点数(包含None节点)
        features_: list
            sklearn.tree.tree_.feature
        thresholds_: list
            sklearn.tree.tree_.threshold
        v_leaf_: int
            sklearn.tree_.feature中表示决策树叶子的值,根据 sklearn 0.17.1 将
            sklearn.tree_.feature中的最后一项的值认为是决策树叶子的值
        paths_: list
            所有决策树交叉项的生成路径

        参考
        ----
        sklearn.tree.DecisionTreeClassifier, sklearn.tree.DecisionTreeRegressor
        """
    def __init__(self,
                 columns,
                 n_decimals=3,
                 v_convert_functions=None,
                 sep=', ',
                 inters_only=False,
                 tree=None,
                 tree_=None,
                 max_depth=5,
                 **kwargs):
        super(RegressionDTDiscretizer, self).__init__(
            columns=columns,
            v_convert_functions=v_convert_functions,
            n_decimals=n_decimals,
            sep=sep,
            inters_only=inters_only,
            classification=False,
            tree=tree,
            tree_=tree_,
            max_depth=max_depth,
            **kwargs)


class GBDTDiscretizer(BaseEstimator,
                         TransformerMixin):
    def __init__(self,
                 columns,
                 n_decimals=6,
                 v_convert_functions=None,
                 sep=', ',
                 inters_only=False,
                 classification=True,
                 **kwargs):
        self.columns = list(columns)
        self.n_decimals = n_decimals
        self.v_convert_functions = v_convert_functions
        self.sep = sep
        self.inters_only = inters_only
        self.classification = classification
        self.kwargs = kwargs
        self.gbdt_inter_feature_names_ = []
        self.unique_inter_feature_indexes_ = []
        print(u'\nGBDTDiscretizer友情提醒：\n'
              u'n_decimals的设置在GBDTDiscretizer中会影响对重复特征的删除操作，\n'
              u'过小会使针对同一特征threshold差距很小的两个不同条件被认为是相同的。\n'
              u'故建议多取几个值看看效果')

    def fit(self, X, y):
        """拟合GBDTDiscretizer，生成交叉项特征名称

        :param X: array-like, shape = [n_samples, n_features]
            拟合GBDT使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :param y: array-like, shape = [n_samples]
            目标矩阵
        :return: self
        """
        # 训练前重置该discretizer
        self._reset()

        # 使用gbdt进行拟合
        if self.classification:
            gbdt = GradientBoostingClassifier(**self.kwargs)
        else:
            gbdt = GradientBoostingRegressor(**self.kwargs)
        gbdt.fit(X, y)

        # 提取gbdt中的所有决策树
        trees = list(gbdt.estimators_.flatten())
        tree_s = [(tree.tree_.feature, tree.tree_.threshold) for tree in trees]

        # 根据提取出的所有决策树生成对应DTDiscretizer
        self.tree_discretizers_ = [RegressionDTDiscretizer(columns=self.columns,
                                                      n_decimals=self.n_decimals,
                                                      v_convert_functions=self.v_convert_functions,
                                                      sep=self.sep,
                                                      inters_only=True,
                                                      tree_=tree_).fit(X, y)
                              for tree_ in tree_s]

        # 记录gbdt中所有单棵决策树离散化的节点名称
        for discretizer in self.tree_discretizers_:
            self.gbdt_inter_feature_names_.extend(discretizer.get_column_names())

        # 记录下所有非重复节点在self.gbdt_inter_feature_names_中的index
        inters_set = set(self.gbdt_inter_feature_names_)
        for fea_tmp in inters_set:
            self.unique_inter_feature_indexes_.append(self.gbdt_inter_feature_names_.index(fea_tmp))
        return self

    def transform(self, X):
        """使用GBDTDiscretizer对数据集进行转化

        :param X: array-like [n_samples, n_features]
            拟合GBDT使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :return: np.array shape = [n_samples, (n_features+len(self.unique_inter_feature_indexes_))]
            对源数据进行转化后的数据集
        """
        if self.tree_discretizers_ is None:
            print(u'请先对数据集使用fit函数,或直接调用fit_transform函数')
            return None

        re_X = []
        for discretizer in self.tree_discretizers_:
            re_X.append(discretizer.transform(X))
        re_X = np.concatenate(tuple(re_X), axis=1)
        re_X = re_X[:, self.unique_inter_feature_indexes_]

        if self.inters_only:
            return re_X
        else:
            return np.concatenate((X, re_X), axis=1)

    def fit_transform(self, X, y):
        """GBDTDiscretizer先fit数据集X再对X进行transform,跟先调用fit再调用transform的结果无差别。
        然而不建议对测试集使用该函数。

        :param X: array-like [n_samples, n_features]
            拟合GBDT使用的数据集,建议对X中的名义变量使用独热编码后再传参
        :param y: array-like [n_samples]
            目标矩阵
        :return: np.array shape = [n_samples, (n_features+len(self.unique_inter_feature_indexes_))]
            对源数据进行转化后的数据集
        """
        self.fit(X, y)
        return self.transform(X)

    def get_column_names(self):
        """获得新生成数据节各列的列名

        :return: list 新数据集各列列名
        """
        if self.inters_only:
            return list(np.array(self.gbdt_inter_feature_names_)[self.unique_inter_feature_indexes_])
        else:
            return self.columns + list(np.array(self.gbdt_inter_feature_names_)[self.unique_inter_feature_indexes_])

    def _reset(self):
        """重置该discretizer

        :return: Modified function
        """
        self.tree_discretizers_ = None
        self.gbdt_inter_feature_names_ = []
        self.unique_inter_feature_indexes_ = []

