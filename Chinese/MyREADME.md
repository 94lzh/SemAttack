SemAttack
以下都说的是Chinese文件夹下的文件。
1.参考requirements.txt安装相应依赖,python推荐用3.9，环境最好用conda创建
2.使用myTrain.py进行微调bert模型以能够分类欺诈数据集,由于硬件资源比较差，我采用1000条训练数据train_1000.csv和100条验证数据val_100.csv来训练以加快速度，但训练出来的模型data/bert/bert/model/pth可以在完整2500多条测试数据test.csv中达到90%以上的准确率，可以使用test.py脚本进行验证
3.构建词嵌入向量空间。我参考"Visualizing and Measuring the Geometry of BERT"对应仓库的代码改进后进行训练，不过是在那个仓库里跑出来的，脚本复制到Chinese文件夹下，不够不能确保环境不冲突，建议克隆仓库后在context-atlas文件夹下使用该脚本和训练数据集train.csv运行得到词嵌入向量
4.使用csvbasepkl.py先将测试数据集test.csv转换成fraud_base_data.pkl,然后分别依次运行get_FC.py、get_FT.py、get_FK.py，最后得到all_fraud_base_data.pkl
5.最后使用修改后的MyAttack.py运行进行攻击
举例来说，可以这样：
python MyAttack.py \
  --function cluster \
  --load path_to_pretrained_model \
  --test-model path_to_pretrained_model \
  --test-data path_to_dataset_with_embedding_space \
  --untargeted
