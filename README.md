大数据环境下文本情感分析算法的规模适配研究：以Twitter为数据源   [pdf](./大数据环境下文本情感分析算法的规模适配研究-以Twitter为数据源.pdf)  [web](http://kns.cnki.net/kcms/detail/11.1541.G2.20190220.0953.003.html)  [数据集](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
===


摘 要
--------
<b>[目的/意义] </b> 以大数据环境下的文本情感分析这一特定任务为目的，对规模适配问题进行研究，为情报学领域研究人员进行大数据环境下数据分析时，实现效率和成本的最优选择提供借鉴。
<b>[方法/过程] </b> 采用斯坦福大学Sentiment140数据集，在对传统情感分析算法分析的基础上，提出了5种面向大数据的文本情感分析算法，检验各种算法在不同环境和数据规模下的适配效果，从准确性、可扩展性和效率等方面进行实证比较研究。
<b>[结果/结论] </b> 实验结果显示，本文所搭建的集群具有良好的运行效率、正确性以及可扩展性，Spark集群在处理海量文本情感分析数据时更具有效率优势，且在数据规模越大的情况下，效率优势越明显；在资源利用方面，随着节点数和核数的增加，集群的整体运行效率变化显著，配置5个4核4G内存的从节点，能够实现在高效完成分类任务。

实 验
---------
- 实验一：比较5种分类算法在传统、单节点Spark和集群Spark运行效率
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eoffrjvyj20j20aywey.jpg" 
style="zoom:60%" align=center  />

- 实验二：比较5种分类算法在传统、单节点Spark和集群Spark中的正确性
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eol58tyvj20iv0a974m.jpg" 
style="zoom:60%" align=center />

- 实验三：比较集群Spark中运行时间随从节点核数变化情况，以评估集群的可扩展性
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eom2opvbj20jm0a5t95.jpg" 
style="zoom:60%" align=center />

- 实验四：比较集群Spark中运行时间随从节点个数变化情况，图5展示了算法运行时间随从节点个数变化
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eonq4378j20iw0abjrt.jpg" 
style="zoom:60%" align=center />

- 图6：展示了单节点下不同算法的运行时间情况
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eopgiofgj20j20a4wez.jpg" 
style="zoom:60%" align=center />

- 图7：展示了5个从节点下不同算法的运行时间情况
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eopureoqj20iw0afwez.jpg" 
style="zoom:60%" align=center />

- 图8：展示了算法运行加速比随数据规模变化情况
<img src="https://ws1.sinaimg.cn/mw690/e669e01fly1g2eoral1znj20iv0a9jrw.jpg" 
style="zoom:60%" align=center />


环 境
------------
- 服务器平台配置信息
<img src="https://ws1.sinaimg.cn/large/e669e01fly1g2eoz64xwij20r408e3z3.jpg"
style="zoom:60%" align=center />

参 考
--------
> **余传明,原赛,王峰,安璐.大数据环境下文本情感分析算法的规模适配研究:以Twitter为数据源[J].图书情报工作:1-10.**