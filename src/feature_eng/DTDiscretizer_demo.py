import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from feature_eng.tree_based.dtree_discretizer import ClassificationDTDiscretizer, GBDTDiscretizer

## 测试时需修改该路径
path = '/Users/ZacharyHE/Documents/Work/svn/personal/currently_working/feature_eng/SFPD_Incidents_September_2016.csv'

# 生成测试用数据集:生成Y值,violent;删去有过多类的名义变量,以方便查验;
# 提取时间特征的小时部分;将名义变量转成独热编码;
# 最终使用X,y作为测试的数据集,X为pd.DataFrame
data = pd.read_csv(path, header=0)
data.set_index(keys=['PdId'], inplace=True)
data['violent'] = data['Category'].isin(['ASSAULT', 'ROBBERY', 'SEX OFFENSES, FORCIBLE', 'KIDNAPPING']) \
                  | data['Descript'].isin(['GRAND THEFT PURSESNATCH', 'ATTEMPTED GRAND THEFT PURSESNATCH'])
data.drop(labels=['Category', 'Descript', 'Address', 'Location', 'Date', 'IncidntNum'], axis=1, inplace=True)
data['Time'] = data.Time.str[:2].astype(int)
data_types_dict = {k.name:v for k, v in
                   data.columns.to_series().groupby(data.dtypes).groups.items()}
data = pd.get_dummies(data, prefix=data_types_dict['object'], prefix_sep=':', dummy_na=True)
y = data['violent']
X = data.drop(labels =['violent'], axis=1)


######## 使用示例 ########
# 定义决策树的参数(测试时可修改)
params = {'max_depth':5}

#### 交叉项生成示例 ####
## 决策树
dtd = ClassificationDTDiscretizer(X.columns, 3, **params)
new_x = dtd.fit_transform(X, y)
new_names = dtd.get_column_names()
## GBDT
gbdis = GBDTDiscretizer(columns=X.columns, n_decimals=6)
new_X_gbdt = gbdis.fit_transform(X, y)
new_names_gbdt = gbdis.get_column_names()

#### Pipeline使用示例 ####
sc = StandardScaler()
dtd = ClassificationDTDiscretizer(X.columns, 3, **params)
lr = LogisticRegression()
clf = Pipeline([('dtd', dtd), ('sc', sc), ('lr', lr)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)
lr_acc = (lr_predict == y_test).mean()
clf_acc = (clf_predict == y_test).mean()
col_names = clf.steps[0][1].get_column_names()

#### 使用GridSearchCV对clf进行正则 ####
parameters = {'lr__C' : [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]}
grid_cv = GridSearchCV(clf, parameters, scoring='accuracy', cv=5)
grid_cv.fit(X_train, y_train)
grid_predict = grid_cv.best_estimator_.predict(X_test)
grid_acc = (grid_predict == y_test).mean()

## 输出结果 ##
print(u"单逻辑回归模型准确率为%.2f%%" % (lr_acc*100))
print(u"使用ClassificationDTDiscretizer生成新变量后（未做特征选择）模型准确率为%.2f%%" % (clf_acc*100))
print(u"对模型进行正则后,正则系数为%.4f, 准确率为%.2f%%" % (grid_cv.best_params_['lr__C'], clf_acc*100))
print(u"完成测试..")


# ######## 为决策树作图以供查验 ######## (uncomment this block if you want to check the correctness)
# tree = DecisionTreeClassifier(**params)
# tree.fit(X, y)
# # 给决策树作图,需cd到dtree.dot所在文件夹,键入 dot -Tpng dtree.dot -o dtree.png 命令可查看图像
# dir_path = 'the/path/of/the/directory/you/want/to/put/dtree.dot' # 查验时需做修改
# dotfile = open(os.path.join(dir_path, 'dtree.dot'), 'w')
# export_graphviz(tree, out_file=dotfile, feature_names=X.columns)
# dotfile.close()
