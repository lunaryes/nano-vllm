## 为什么改
当前使用的调度方法是prefill无条件优先，只有prefill完成才能进入decode，这就意味着在多序列请求下，首次TTFT会比较大。

## 改的是什么
改为使用vllm-V1所使用的FCFS+token budget的策略。prefill和decode混合运行以提高TTFT。

## 系统现在长什么样 


## 哪里会被打破
## 接口怎么定
## 状态怎么走
## 算法和代码